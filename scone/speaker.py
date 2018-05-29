import argparse
import json
import random
import time
from collections import Counter, namedtuple

import dynet as dy
import numpy as np
import os
import scipy
import tqdm

import scone.corpus
import util
from models import run_lstm, OneHotVocabEmbeddings
from util import GlorotInitializer, softmax

import metrics.bleu

import sys

InferenceState = namedtuple('InferenceState', 'prev_inference_state, lstm_state, lstm_h, prev_token, prev_token_index, token_count, action_count, log_probs')

EPS = 1e-8

def blank_inference_state(action_count):
    return InferenceState(prev_inference_state=None,
                          lstm_state=None,
                          lstm_h=None,
                          prev_token=None,
                          prev_token_index=None,
                          token_count=0,
                          action_count=action_count,
                          log_probs=None)

def backchain(last_inference_state):
    tokens = []
    inf_state = last_inference_state
    while inf_state is not None:
        tokens.append(inf_state.prev_token)
        inf_state = inf_state.prev_inference_state
    return list(reversed(tokens))


def _unpack_beam_item(beam_item, BOS, EOS):
    score, inf_states = beam_item
    all_tokens = [backchain(inf_state) for inf_state in inf_states]
    assert(util.all_equal(all_tokens))
    tokens_with_markers = all_tokens[0]

    assert tokens_with_markers[0] == BOS
    assert tokens_with_markers[-1] == EOS

    return tokens_with_markers[1:-1], score

def ensemble_speak(models, ensemble_average_probs, states, actions, sample=False, sample_alpha=1.0, random_state=None, gold_utterances=None, max_utterance_length=35, length_constraints=None):
    assert len(states) == len(actions) + 1

    if length_constraints:
        assert len(length_constraints) == len(actions)
        assert all(l <= max_utterance_length for l in length_constraints)

    assert all(m.vocab_embeddings.int_to_word == models[0].vocab_embeddings.int_to_word for m in models[1:])
    EOS_INDEX = models[0].vocab_embeddings.EOS_INDEX

    BOS = models[0].vocab_embeddings.BOS
    EOS = models[0].vocab_embeddings.EOS


    if gold_utterances is not None:
        assert len(gold_utterances) == len(actions)
        if length_constraints is not None:
            for utt, lc in zip(gold_utterances, length_constraints):
                assert len(utt) == lc

    initialize_decoder_fns, predict_word_fns, choose_word_fns = zip(*[
        model.state_machine(states, actions, testing=True) for model in models
    ])

    pred_utterances_and_scores = []

    for i in range(len(actions)):
        token_count = 0
        gold_utterance = gold_utterances[i] if gold_utterances is not None else None

        score = 0
        inference_states = [init_decoder(blank_inference_state(i))
                            for init_decoder in initialize_decoder_fns]
        while True:
            inference_states = [predict(inf_state)
                                for predict, inf_state in zip(predict_word_fns, inference_states)]
            log_probs = np.mean([inf_state.log_probs.npvalue()
                                 for inf_state in inference_states], axis=0)

            if gold_utterance is not None:
                if token_count >= len(gold_utterance):
                    token_index = EOS_INDEX
                else:
                    token_index = models[0].vocab_embeddings.index_word(gold_utterance[token_count])
            #elif token_count >= max_utterance_length or (length_constraint and token_count >= length_constraint):
            elif token_count >= max_utterance_length or (length_constraints and token_count >= length_constraints[i]):
                token_index = EOS_INDEX
            else:
                if length_constraints:
                    assert token_count < length_constraints[i]
                    log_probs[EOS_INDEX] = -np.inf
                if sample:
                    if random_state is not None:
                        r = random_state.random()
                    else:
                        r = random.random()
                    if sample_alpha != 1.0:
                        smoothed_dist_v = softmax(log_probs * sample_alpha)
                    else:
                        smoothed_dist_v = softmax(log_probs)
                    token_index = smoothed_dist_v.cumsum().searchsorted(r)
                else:
                    token_index = log_probs.argmax()

            inference_states = [choose(inf_state, token_index)
                                for choose, inf_state in zip(choose_word_fns, inference_states)]
            score += log_probs[token_index]

            if token_index == EOS_INDEX:
                break
            token_count += 1
        pred_utterances_and_scores.append(_unpack_beam_item((score, inference_states), BOS=BOS, EOS=EOS))

    return pred_utterances_and_scores


def beam_ensemble_speak(beam_size, models, ensemble_average_probs, states, actions,
                        return_beam=False, max_utterance_length=35, length_constraints=None):

    assert len(states) == len(actions) + 1
    if length_constraints:
        assert len(length_constraints) == len(actions)
        assert all(lc <= max_utterance_length for lc in length_constraints)
    assert all(m.vocab_embeddings.int_to_word == models[0].vocab_embeddings.int_to_word for m in models[1:])
    EOS_INDEX = models[0].vocab_embeddings.EOS_INDEX

    BOS = models[0].vocab_embeddings.BOS
    EOS = models[0].vocab_embeddings.EOS

    initialize_decoder_fns, predict_word_fns, choose_word_fns = zip(*[
        model.state_machine(states, actions, testing=True) for model in models
    ])

    utterance_beams = []

    for i in range(len(actions)):
        token_count = 0
        completed = []

        beam = [(0, [init_decoder(blank_inference_state(i)) for init_decoder in initialize_decoder_fns])]

        while len(completed) < beam_size:

            successors = []
            for (beam_item_score, inference_states) in beam:
                inference_states = [predict(inf_state)
                                    for predict, inf_state in zip(predict_word_fns, inference_states)]

                lps = [inf_state.log_probs.npvalue() for inf_state in inference_states]
                if ensemble_average_probs:
                    log_probs = scipy.misc.logsumexp(lps, axis=0) - np.log(len(models))
                else:
                    log_probs = np.mean(lps, axis=0)

                #num_successors = min(beam_size, len(scored_actions[scored_actions > 0]))
                num_successors = min(beam_size, len(log_probs))
                if token_count >= max_utterance_length or (length_constraints and token_count >= length_constraints[i]):
                    successor_token_ixs = [EOS_INDEX]
                else:
                    successor_token_ixs = np.argpartition(log_probs, -num_successors)[-num_successors:]

                for successor_number, token_index in enumerate(reversed(successor_token_ixs)):
                    if length_constraints and token_count < length_constraints[i] and token_index == EOS_INDEX:
                        continue
                    successor_inference_states = [choose(inf_state, token_index)
                                                  for choose, inf_state in zip(choose_word_fns, inference_states)]
                    marginal_log_prob = log_probs[token_index]
                    successor_score = beam_item_score + marginal_log_prob

                    complete = token_index == EOS_INDEX
                    successors.append((complete, successor_score, successor_inference_states))

            successors = sorted(successors, key=lambda t: t[1], reverse=True)[:beam_size]
            beam = []
            for (complete, score, inf_states) in successors:
                if complete:
                    completed.append((score, inf_states))
                else:
                    beam.append((score, inf_states))

            token_count += 1

        completed = sorted(completed, key=lambda t: t[0], reverse=True)[:beam_size]

        utterance_beams.append([_unpack_beam_item(bi, BOS=BOS, EOS=EOS) for bi in completed])

    if return_beam:
        return utterance_beams
    else:
        return [beam[0] for beam in utterance_beams]


class Speaker(dy.Saveable):
    def __init__(self,
                 model,
                 corpus,
                 vocab_embeddings,
                 embedded_y_dim,
                 enc_state_dim,
                 dec_state_dim,
                 random_seed=None,
                 bidi=False,
                 # attention=None,
                 # attention_dim=50,
                 dropout=None,
                 # ablate_world=False,
                 # ablate_language=False,
                 # feed_actions_to_decoder=False
                 ):
        # model: dynet.Model
        # vocab_embeddings: VocabEmbeddings
        self.corpus = corpus
        self.vocab_embeddings = vocab_embeddings

        self.embedded_y_dim = embedded_y_dim

        self.enc_state_dim = enc_state_dim
        self.dec_state_dim = dec_state_dim

        init = GlorotInitializer(random_seed)

        # self.enc_input_dim = self.vocab_embeddings.embedding_dim
        self.enc_input_dim = self.corpus.ACTION_DIM + self.embedded_y_dim + self.corpus.ACTION_IN_STATE_CONTEXT_DIM

        self.z_dim = self.enc_state_dim * (2 if bidi else 1) + self.enc_input_dim
        # if attention:
        #     self.z_dim += self.vocab_embeddings.embedding_dim

        # self.dec_input_dim = (self.z_dim * self.action_layer.num_attention_heads) + self.embedded_y_dim
        self.dec_input_dim = self.vocab_embeddings.embedding_dim + self.z_dim


        # if feed_actions_to_decoder:
        #     self.dec_input_dim += self.corpus.ACTION_DIM

        self.random_seed = random_seed

        self.bidi = bidi
        # self.attention = attention
        # self.attention_dim = attention_dim

        self.dropout = dropout

        # self.ablate_world = ablate_world
        # self.ablate_language = ablate_language

        # self.feed_actions_to_decoder = feed_actions_to_decoder


        self.enc_fwd_lstm = dy.LSTMBuilder(1,
                                           self.enc_input_dim,
                                           self.enc_state_dim,
                                           model)

        if self.bidi:
            self.enc_bwd_lstm = dy.LSTMBuilder(1,
                                               self.enc_input_dim,
                                               self.enc_state_dim,
                                               model)

        self.dec_lstm = dy.LSTMBuilder(1,
                                       self.dec_input_dim,
                                       self.dec_state_dim,
                                       model)

        self.p_L0 = model.parameters_from_numpy(init.initialize((self.vocab_embeddings.vocab_size, self.dec_state_dim + self.z_dim)))
        self.p_E_state = model.parameters_from_numpy(init.initialize((self.embedded_y_dim, self.corpus.STATE_DIM)))
        # self.p_E_action = model.parameters_from_numpy(init.initialize((self.embedded_y_dim, self.corpus.ACTION_DIM)))


    def get_components(self):
        components = [self.enc_fwd_lstm, self.dec_lstm, self.p_E_state, self.p_L0, self.vocab_embeddings]
        if self.bidi:
            components.append(self.enc_bwd_lstm)
        return tuple(components)

    def restore_components(self, components):
        k = 5
        self.enc_fwd_lstm, self.dec_lstm, self.p_E_state, self.p_L0, self.vocab_embeddings = components[:k]
        if self.bidi:
            self.enc_bwd_lstm = components[k]
            k += 1
        assert k == len(components)

    def encode_inputs(self, vecs, apply_dropout):
        if apply_dropout and self.dropout:
            self.enc_fwd_lstm.set_dropout(self.dropout)
            if self.bidi:
                self.enc_bwd_lstm.set_dropout(self.dropout)
        else:
            self.enc_fwd_lstm.disable_dropout()
            if self.bidi:
                self.enc_bwd_lstm.disable_dropout()

        fwd_states, fwd_h = run_lstm(self.enc_fwd_lstm.initial_state(), vecs)
        vecs_rev = list(reversed(vecs))
        if self.bidi:
            bwd_states, bwd_h = run_lstm(self.enc_bwd_lstm.initial_state(), vecs_rev)
            vec_outputs = [dy.concatenate(list(p))
                           for p in zip(fwd_states, list(reversed(bwd_states)))]
            return vec_outputs, (fwd_states[-1], bwd_states[-1])
        else:
            vec_outputs = fwd_states
            return vec_outputs, (fwd_states[-1],)


    def state_machine(self, states, actions, testing=True):
        # add params to the computation graph
        L0 = dy.parameter(self.p_L0)
        E_state = dy.parameter(self.p_E_state)
        #E_action = dy.parameter(self.p_E_action)

        assert len(states) == len(actions) + 1

        embedded_states = [dy.inputVector(self.corpus.embed_state(state)) for state in states]
        embedded_actions = [dy.inputVector(self.corpus.embed_action(action)) for action in actions]
        embedded_actions_in_sc = []
        for i in range(len(actions)):
            v = self.corpus.embed_action_in_state_context(actions[i], states[i], states[i+1], states[:i], actions[:i])
            embedded_actions_in_sc.append(dy.inputVector(v))
        # embedded_actions_in_sc = [dy.inputVector(self.corpus.embed_action_in_state_context(action, state_before, state_after))
        #                                          for action, state_before, state_after in zip(actions, states, states[1:])]
        embedded_actions.append(dy.inputVector(np.zeros(self.corpus.ACTION_DIM)))
        embedded_actions_in_sc.append(dy.inputVector(np.zeros(self.corpus.ACTION_IN_STATE_CONTEXT_DIM)))

        assert len(embedded_states) == len(embedded_actions) == len(embedded_actions_in_sc)

        # todo: consider embedding actions as well?
        embedded_state_actions = [dy.concatenate([E_state * s, a, asc])
                                  for (s, a, asc) in zip(embedded_states, embedded_actions, embedded_actions_in_sc)]

        # apply dropout if not testing
        encoded_state_actions, _ = self.encode_inputs(embedded_state_actions, not testing)

        zs = [dy.concatenate([emb, enc]) for (emb, enc) in zip(embedded_state_actions, encoded_state_actions)]

        if not testing and self.dropout:
            self.dec_lstm.set_dropout(self.dropout)
        else:
            self.dec_lstm.disable_dropout()

        def initialize_decoder(state):
            return state._replace(
                lstm_state=self.dec_lstm.initial_state(),
                lstm_h=[dy.vecInput(self.dec_state_dim)],
                prev_token=self.vocab_embeddings.BOS,
                prev_token_index=self.vocab_embeddings.BOS_INDEX,
                token_count=0,
            )

        def predict_word(state):
            # update state lstm_state and h with the attended input, and
            # compute the action distribution log_probs for the state
            assert state.action_count < len(actions) # len(zs) should be == to len(states) == len(actions) + 1

            z = zs[state.action_count]

            dec_input = dy.concatenate([self.vocab_embeddings.embed_word_index(state.prev_token_index), z])

            new_lstm_state = state.lstm_state.add_input(dec_input)
            new_lstm_h = new_lstm_state.h()

            out = new_lstm_state.output()
            q = L0 * dy.concatenate([out, z])

            new_log_probs = dy.log_softmax(q)

            return state._replace(lstm_state=new_lstm_state,
                                  lstm_h=new_lstm_h,
                                  log_probs=new_log_probs)

        def choose_word(inference_state, token_index):
            new_token_count = inference_state.token_count + 1
            token = self.vocab_embeddings.int_to_word[token_index]

            new_action_count = inference_state.action_count
            if token_index == self.vocab_embeddings.EOS_INDEX:
                new_action_count += 1

            return inference_state._replace(
                prev_inference_state=inference_state,
                prev_token=token,
                prev_token_index=token_index,
                token_count=new_token_count,
                action_count=new_action_count,
                log_probs=None,
            )

        return initialize_decoder, predict_word, choose_word


    def loss(self, states, actions, utterances, testing=False, loss_per_action=False):
        assert len(states) - 1 == len(actions) == len(utterances)

        initialize_decoder, predict_word, choose_word = self.state_machine(states, actions, testing=testing)

        losses_per_action = []
        for i in range(len(actions)):
            inf_state = initialize_decoder(blank_inference_state(i))
            this_action_losses = []
            for token in utterances[i] + [self.vocab_embeddings.EOS]:
                inf_state = predict_word(inf_state)
                token_index = self.vocab_embeddings.index_word(token)
                loss = -dy.pick(inf_state.log_probs, token_index)
                this_action_losses.append(loss)
                inf_state = choose_word(inf_state, token_index)
            losses_per_action.append(this_action_losses)

        assert inf_state.action_count == len(actions)

        if loss_per_action:
            return [dy.esum(losses) for losses in losses_per_action]
        else:
            return dy.esum([loss for losses in losses_per_action for loss in losses])

    # TODO: decoders, training


def fmt_state(state):
    return ' '.join("%d:%s" % (i+1, contents)
                    for i, contents in enumerate(state))


# def compare_prediction(agent, instance, predicted_states, predicted_actions, attention_dists):
#     def match_print(x, y):
#         print("%s\t%s\t%s" % ("+" if x == y else "_", x, y))
#
#     def print_utt(utt):
#         print(" \t%s" % ' '.join(s[:6].ljust(6) for s in utt))
#
#     for i in range(len(predicted_actions)):
#         print(' '.join(instance.utterances[i]))
#         match_print(fmt_state(instance.states[i]), fmt_state(predicted_states[i]))
#         match_print(instance.actions[i], predicted_actions[i])
#         match_print(fmt_state(instance.states[i+1]), fmt_state(predicted_states[i+1]))
#         print_utt(instance.utterances[i])
#         for attn_name, attn_dist in zip(agent.action_layer.attention_names, attention_dists[i]):
#             print("%s\t%s" % (attn_name, ' '.join("%0.4f" % x for x in attn_dist.npvalue())))
#         print()
#
#         pass

def in_training_indicator(utterance, train_utterances):
    if train_utterances is None:
        return " "
    if tuple(utterance) in train_utterances:
        return "+"
    else:
        return "-"

def compare_prediction(instance, hyp_utterance_beams, num_actions_to_take=None, train_utterances=None):
    if num_actions_to_take is None:
        num_actions_to_take = len(instance.utterances)

    for action_ix in range(num_actions_to_take):
        print("      \t%s" % str(instance.actions[action_ix]))
        print("      \t%s" % fmt_state(instance.states[action_ix]))
        print("%s ref:\t%s" % (in_training_indicator(instance.utterances[action_ix], train_utterances),
                               ' '.join(instance.utterances[action_ix])))
        print("      \t%s" % fmt_state(instance.states[action_ix+1]))
        for beam_ix, (h, _) in enumerate(hyp_utterance_beams[action_ix]):
            print("%s hyp %2d\t%s" % (in_training_indicator(h, train_utterances), beam_ix, ' '.join(h)))
        print()

def evaluate(instances, pred_uterancess, num_actions_to_take=None, name='', print_results=True):

    # if num_actions_to_take is None:
    #     num_actions_to_take = scone.corpus.NUM_TRANSITIONS

    assert len(instances) == len(pred_uterancess)
    if num_actions_to_take is not None:
        assert all(len(utt) == num_actions_to_take for utt in pred_uterancess)

    ref_utterances = []
    for inst, pred_utterances in zip(instances, pred_uterancess):
        ref_utterances.append(inst.utterances[:len(pred_utterances)])

    assert all(len(ref) == len(pred)
               for ref, pred in zip(ref_utterances, pred_uterancess))

    ref_flat = util.flatten(ref_utterances)
    hyp_flat = util.flatten(pred_uterancess)

    avg_ref_length = np.mean([len(utt) for utt in ref_flat])
    avg_hyp_length = np.mean([len(utt) for utt in hyp_flat])

    bleu, unpenalized_bleu = metrics.bleu.single_bleu(ref_flat, hyp_flat)
    if print_results:
        print("%s bleu:\t%s" % (name, bleu))
        print("%s unpenalized_bleu:\t%s" % (name, unpenalized_bleu))
    return {'bleu': bleu,
            'unpenalized_bleu': unpenalized_bleu,
            'avg_hyp_length': avg_hyp_length,
            'avg_ref_length': avg_ref_length}

def test_on_instances(agents, instances, beam_size=None, num_actions_to_take=None, verbose=False, name='', max_utterance_length=35, train_instances_for_comparison=None, force_length_match=False):
    refs_by_instance = []
    hyps_by_instance = []

    scores = []

    if train_instances_for_comparison:
        train_utterances = set(tuple(utt) for inst in train_instances_for_comparison
                               for utt in inst.utterances)
    else:
        train_utterances = None

    for ix, instance in enumerate(instances):
        states = instance.states
        actions = instance.actions
        ref = instance.utterances

        if force_length_match:
            length_constraints = [len(utt) for utt in ref]
        else:
            length_constraints = None

        if num_actions_to_take is not None:
            states = states[:num_actions_to_take + 1]
            actions = actions[:num_actions_to_take]
            ref = ref[:num_actions_to_take]

        dy.renew_cg()
        if beam_size is None:
            utterance_hyp_scores = ensemble_speak(agents, False, states, actions, max_utterance_length=max_utterance_length, length_constraints=length_constraints)
            hyp_beams = [[t] for t in utterance_hyp_scores]
            hyp, utt_scores = zip(*utterance_hyp_scores)
            score = sum(utt_scores)
        else:
            # list (one item per action) of beams, where each beam item is (utterance, score)
            hyp_beams = beam_ensemble_speak(beam_size, agents, False, states, actions, return_beam=True, max_utterance_length=max_utterance_length, length_constraints=length_constraints)
            hyp, utt_scores = zip(*[beam[0] for beam in hyp_beams])
            score = sum(utt_scores)

        scores.append(score)

        assert len(hyp) == len(ref)
        hyps_by_instance.append(hyp)
        refs_by_instance.append(ref)

        if verbose and ix % 20 == 0:
            print(ix)
            print(instance.id)
            compare_prediction(instance, hyp_beams, num_actions_to_take=num_actions_to_take, train_utterances=train_utterances)

    stats = evaluate(instances, hyps_by_instance,
                     num_actions_to_take=num_actions_to_take, name=name, print_results=True)
    stats['beam_size'] = beam_size
    return stats

def train(args):
    if args.embedded_y_dim is None:
        args.embedded_y_dim = args.enc_state_dim

    if args.save_dir:
        try:
            os.mkdir(args.save_dir)
        except:
            pass

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    corpus, train_instances, dev_instances, test_instances = scone.corpus.load_corpus_and_data(args)

    if args.train_instances_limit:
        train_instances = train_instances[:args.train_instances_limit]

    train_subset_instances = train_instances[:len(train_instances)//10]

    max_utterance_length = max(len(utt) for instance in train_instances + dev_instances
                               for utt in instance.utterances) + 3

    model = dy.Model()

    optimizer = dy.AdamTrainer(model)

    if args.load_dir:
        fname = os.path.join(args.load_dir, args.corpus)
        print("loading from %s" % fname)
        speaker = model.load(fname)[0]
    else:
        vocab_counter = Counter([token for instance in train_instances
                                 for utt in instance.utterances
                                 for token in utt])
        vocab = [t for (t, count) in vocab_counter.items() if count > args.unk_threshold]
        vocab_embeddings = OneHotVocabEmbeddings(model, vocab)
        speaker = Speaker(model,
                        corpus=corpus,
                        vocab_embeddings=vocab_embeddings,
                        embedded_y_dim=args.embedded_y_dim,
                        enc_state_dim=args.enc_state_dim,
                        dec_state_dim=args.dec_state_dim,
                        random_seed=args.random_seed,
                        bidi=args.bidi,
                        dropout=args.dropout)

    train_stats = []
    dev_stats = []
    test_stats = []

    def save_model():
        if args.save_dir:
            fname = os.path.join(args.save_dir, args.corpus)
            print("saving to %s" % fname)
            model.save(fname, [speaker])

    def decode_fn(instances, name, speakers=[speaker]):
        num_actions_to_take = args.eval_num_actions_to_take or None
        if args.beam_size is not None and args.beam_size > 1:
            print("greedy:")
            test_on_instances(speakers, instances, beam_size=None, num_actions_to_take=num_actions_to_take,
                              verbose=args.verbose, name=name+"_greedy", max_utterance_length=max_utterance_length,
                              train_instances_for_comparison=train_instances, force_length_match=args.force_length_match)
            print()
            print("beam: ", args.beam_size)
        results = test_on_instances(speakers, instances,
                                    beam_size=args.beam_size, num_actions_to_take=num_actions_to_take,
                                    verbose=args.verbose, name=name+"_beam", max_utterance_length=max_utterance_length,
                                    train_instances_for_comparison=train_instances, force_length_match=args.force_length_match)
        print()
        return results

    def loss_fn(instances, name, train=False):
        losses = []
        enum = enumerate(instances)
        if train and not args.hide_progress:
            enum = tqdm.tqdm(enum, desc="instance", total=N_instances)
        for i, instance in enum:
            dy.renew_cg()
            loss = speaker.loss(instance.states, instance.actions, instance.utterances, testing=not train)
            loss_val = loss.value()
            # if np.isfinite(loss_val):
            #     print()
            #     print("infinite loss:")
            losses.append(loss_val)
            if train:
                loss.backward()
                optimizer.update()
        if train:
            optimizer.status()
        mean_loss = np.mean(losses)
        print("%s loss: %s" % (name, mean_loss))
        print()
        return {'loss': mean_loss}

    def log_train_decode(epochs):
        print("decoding train")
        name = "%d train_subset" % epochs
        stats = {'epoch': epochs}
        stats.update(decode_fn(train_subset_instances, name))
        stats.update(loss_fn(train_subset_instances, name, train=False))
        print(stats)
        train_stats.append(stats)

    def log_dev_decode(epochs):
        print("decoding dev")
        name = "%d dev" % epochs
        stats = {'epoch': epochs}
        stats.update(decode_fn(dev_instances, name))
        stats.update(loss_fn(dev_instances, name, train=False))
        max_dev_bleu = max(s['bleu'] for s in dev_stats) if dev_stats else 0
        if stats['bleu'] > max_dev_bleu:
            save_model()
            if args.test_decode:
                log_test_decode(epochs)
        print(stats)
        dev_stats.append(stats)

    def log_test_decode(epochs):
        print("decoding test")
        name = "%d test" % epochs
        stats = {'epoch': epochs}
        stats.update(decode_fn(test_instances, name))
        stats.update(loss_fn(test_instances, name, train=False))
        print(stats)
        test_stats.append(stats)

    def save_stats():
        if not args.save_dir:
            return
        def write(stats, name):
            with open(os.path.join(args.save_dir, args.corpus + ".%s.stats" % name), 'w') as f:
                json.dump(stats, f, indent=2)
        write(train_stats, "train_subset")
        write(dev_stats, "dev")
        write(test_stats, "test")

    if args.ensemble_dirs:
        speaker_models = [dy.Model() for _ in args.ensemble_dirs]
        speakers = []
        for model, load_dir in zip(speaker_models, args.ensemble_dirs):
            fname = os.path.join(load_dir, args.corpus)
            print("loading model from %s" % fname)
            speakers.append(model.load(os.path.join(load_dir, args.corpus))[0])

        print("ensemble dev decode:")
        decode_fn(dev_instances, "ensemble dev", speakers=speakers)

        if args.test_decode:
            print("ensemble test decode:")
            decode_fn(test_instances, "ensemble test", speakers=speakers)
        sys.exit(0)

    N_instances = len(train_instances)
    for epoch in range(args.train_epochs):
        print("epoch %d" % epoch)
        random.shuffle(train_instances)
        loss_fn(train_instances, "train", train=True)

        if epoch % args.decode_interval == 0:
            # if epoch % args.decode_interval == 0:
            log_train_decode(epoch)
            log_dev_decode(epoch)
            save_stats()

        # decrease learning rate for sgd
        optimizer.update_epoch()

    if (args.train_epochs - 1) % args.decode_interval != 0:
        # log_train_decode(args.train_epochs)
        log_dev_decode(args.train_epochs - 1)
        save_stats()

    return model, speaker

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir")
    parser.add_argument("--load_dir")
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--decode_interval", type=int, default=5)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--train_instances_limit", type=int)

    parser.add_argument("--eval_num_actions_to_take", type=int)
    parser.add_argument("--beam_size", type=int)

    parser.add_argument("--enc_state_dim", type=int, default=100)
    parser.add_argument("--dec_state_dim", type=int, default=100)
    parser.add_argument("--embedded_y_dim", type=int)
    parser.add_argument("--bidi", action='store_true')
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--random_seed", type=int, default=1)

    parser.add_argument("--unk_threshold", type=int, default=1)  # TODO: up this?

    parser.add_argument("--force_length_match", action='store_true')

    parser.add_argument("--test_decode", action='store_true')

    parser.add_argument("--ensemble_dirs", nargs="+")

    parser.add_argument("--hide_progress", action='store_true')

    scone.corpus.add_corpus_args(parser)

    return parser

if __name__ == "__main__":
    util.run(make_parser(), train)
