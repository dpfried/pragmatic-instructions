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
from models import GlobalAttention, run_lstm, OneHotVocabEmbeddings
from util import GlorotInitializer, softmax

import sys

InferenceState = namedtuple('InferenceState', 'prev_inference_state, lstm_state, lstm_h, current_world_state, last_action, action_count, log_probs, possible_actions, attention_dist')

EPS = 1e-8

RANDOMIZED_BEAM_EPSILON = 0.15

BEAM_EXPLORATION_METHODS = ['beam', 'randomized_beam_code', 'randomized_beam_paper']

EXPLORATION_METHODS = BEAM_EXPLORATION_METHODS


def blank_inference_state(current_world_state=None):
    return InferenceState(prev_inference_state=None,
                          lstm_state=None,
                          lstm_h=None,
                          current_world_state=current_world_state,
                          last_action=None,
                          action_count=0,
                          log_probs=None,
                          possible_actions=None,
                          attention_dist=None)

def backchain(last_inference_state):
    states = []
    actions = []
    inf_state = last_inference_state
    attention_dists = []
    while inf_state is not None:
        states.append(inf_state.current_world_state)
        if inf_state.last_action is not None:
            actions.append(inf_state.last_action)
        if inf_state.attention_dist is not None:
            attention_dists.append(inf_state.attention_dist)
        inf_state = inf_state.prev_inference_state
    return list(reversed(states)), list(reversed(actions)), list(reversed(attention_dists))

def _unpack_beam_item(beam_item):
    score, inf_states = beam_item
    all_states, all_actions, attention_dists = zip(*[backchain(inf_state) for inf_state in inf_states])
    assert(util.all_equal(all_states))
    assert(util.all_equal(all_actions))

    return all_states[0], all_actions[0], score, attention_dists[0]

def dynet_logsumexp(xs):
    emacs = dy.emax(xs)
    return dy.log(dy.esum([dy.exp(x - emacs) for x in xs])) + emacs

def ensemble_follow(models, ensemble_average_probs, initial_state, utterances, num_actions_to_take=None, sample=False, sample_alpha=1.0, random_state=None, gold_actions=None, predict_invalid=False, testing=True):
    if num_actions_to_take is None:
        num_actions_to_take = len(utterances)
    assert num_actions_to_take <= len(utterances)

    initialize_decoder_fns, observe_fns, act_fns = zip(*[
        model.state_machine(testing=testing, constrain_actions=True, predict_invalid=predict_invalid) for model in models
    ])

    score = dy.scalarInput(0)
    # score_v = 0.0
    inference_states = [init_decoder(blank_inference_state(initial_state)) for init_decoder in initialize_decoder_fns]

    assert all(m.corpus.dataset_name() == models[0].corpus.dataset_name() for m in models[1:])
    if predict_invalid:
        assert all(m.predict_invalid for m in models)

    corpus = models[0].corpus

    if gold_actions is not None:
        assert len(gold_actions) == num_actions_to_take

    for actions_taken in range(num_actions_to_take):
        inference_states = [observe(inf_state, utterances[actions_taken])
                            for observe, inf_state in zip(observe_fns, inference_states)]

        lps = [inf_state.log_probs for inf_state in inference_states]

        if len(lps) == 1:
            action_log_probs = lps[0]
        elif ensemble_average_probs:
            #assert np.allclose(lse.npvalue(), scipy.misc.logsumexp(lps, axis=0))
            lse = dynet_logsumexp(lps)
            action_log_probs = lse - dy.log(dy.scalarInput(len(models)))
            #action_log_probs = scipy.misc.logsumexp(lps, axis=0) - np.log(len(models))
        else:
            action_log_probs = dy.esum(lps) * dy.scalarInput(1.0 / len(models))
            #assert np.allclose(action_log_probs.npvalue(), np.mean(lps, axis=0))
            #action_log_probs = np.mean(lps, axis=0)
        action_log_probs_v = action_log_probs.npvalue()

        if gold_actions is not None:
            action = gold_actions[actions_taken]
            action_ix = corpus.ACTIONS_TO_INDEX[action]
        else:
            if sample:
                smoothed_probs = softmax(action_log_probs_v * sample_alpha)
                if random_state is not None:
                    r = random_state.random()
                else:
                    r = random.random()
                action_ix = None
                for ix, p in enumerate(smoothed_probs):
                    r -= p
                    if r < 0.0:
                        action_ix = ix
                        break
                assert action_ix is not None
            else:
                action_ix = np.argmax(action_log_probs_v)
            if action_ix == corpus.INVALID_INDEX:
                action = scone.corpus.INVALID
            else:
                action = corpus.ACTIONS[action_ix]
        inference_states = [act(inf_state, action) for act, inf_state in zip(act_fns, inference_states)]
        score += action_log_probs[int(action_ix)]
        # check here rather than at beginning of loop for consistency with beam search,
        # and since we shouldn't ever have a starting state with no valid actions
        if all(not inf_state.possible_actions
               for inf_state in inference_states):
            break
        # score_v += action_log_probs_v[action_ix]
        if action == scone.corpus.INVALID:
            assert predict_invalid
            break

    # states, actions, score, attention_dists
    # return _unpack_beam_item((score_v, inference_states))
    return _unpack_beam_item((score, inference_states))

def eps_greedy_sample_code(scores, num_to_sample, epsilon, possible_action_indices):
    # https://github.com/kelvinguu/lang2program/blob/dd4eb8439d29f0f72dd057946287551ed0f046a3/strongsup/utils.py#L13
    indices_descending = list(sorted(range(len(scores)), key=lambda ix: scores[ix], reverse=True))
    indices_descending = [ix for ix in indices_descending if ix in possible_action_indices]
    if len(indices_descending) <= num_to_sample:
        return indices_descending
    if epsilon == 0:
        return indices_descending[:num_to_sample]
    sample = []
    index_choices = list(range(len(indices_descending)))
    for i in range(num_to_sample):
        if random.random() <= epsilon or not i in index_choices:
            choice_index = random.choice(index_choices)
        else:
            choice_index = i
        sample.append(indices_descending[choice_index])
        index_choices.remove(choice_index)
    return sample

def eps_greedy_sample_paper(scores, num_to_sample, epsilon, possible_action_indices):
    indices_descending = list(sorted(range(len(scores)), key=lambda ix: scores[ix], reverse=True))
    indices_descending = [ix for ix in indices_descending if ix in possible_action_indices]
    if len(indices_descending) <= num_to_sample:
        return indices_descending
    if epsilon == 0:
        return indices_descending[:num_to_sample]
    sample = []
    while len(sample) < num_to_sample:
        assert indices_descending
        if random.random() <= epsilon:
            choice_index = random.choice(indices_descending)
        else:
            choice_index = indices_descending[0]
        sample.append(choice_index)
        indices_descending.remove(choice_index)
    return sample

def beam_ensemble_follow(beam_size, models, ensemble_average_probs, initial_state, utterances,
                         num_actions_to_take=None, return_beam=False, predict_invalid=False, testing=True,
                         exploration_method='beam', randomized_beam_epsilon=RANDOMIZED_BEAM_EPSILON):
    if num_actions_to_take is None:
        num_actions_to_take = len(utterances)
    assert num_actions_to_take <= len(utterances)

    initialize_decoder_fns, observe_fns, act_fns = zip(*[
        model.state_machine(testing=testing, constrain_actions=True, predict_invalid=predict_invalid) for model in models
    ])

    beam = [(dy.scalarInput(0), [init_decoder(blank_inference_state(initial_state))
                                 for init_decoder in initialize_decoder_fns])]

    assert all(m.corpus.dataset_name() == models[0].corpus.dataset_name() for m in models[1:])
    if predict_invalid:
        assert all(m.predict_invalid for m in models)
    corpus = models[0].corpus

    completed = []

    for actions_taken in range(num_actions_to_take):
        # while len(completed) < beam_size and state_count <= max_len:
        # (complete, score, inf_states)
        successors = []
        for (beam_item_score, inference_states) in beam:
            inference_states = [observe(inf_state, utterances[actions_taken]) for observe, inf_state in zip(observe_fns, inference_states)]

            # lps = [inf_state.log_probs.npvalue() for inf_state in inference_states]
            lps = [inf_state.log_probs for inf_state in inference_states]
            if len(lps) == 1:
                action_log_probs = lps[0]
            elif ensemble_average_probs:
                # action_log_probs = scipy.misc.logsumexp(lps, axis=0) - np.log(len(models))
                lse = dynet_logsumexp(lps)
                action_log_probs = lse - dy.log(dy.scalarInput(len(models)))
            else:
                # action_log_probs = np.mean(lps, axis=0)
                action_log_probs = dy.esum(lps) * dy.scalarInput(1.0 / len(models))
            action_log_probs_v = action_log_probs.npvalue()

            #num_successors = min(beam_size, len(scored_actions[scored_actions > 0]))
            possible_actions = inference_states[0].possible_actions
            possible_action_indices = sorted([corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])
            num_successors = min(beam_size, len(possible_actions))
            if exploration_method == 'randomized_beam_code':
                successor_action_ixs = eps_greedy_sample_code(action_log_probs_v, num_successors, randomized_beam_epsilon, possible_action_indices)
            elif exploration_method == 'randomized_beam_paper':
                successor_action_ixs = eps_greedy_sample_paper(action_log_probs_v, num_successors, randomized_beam_epsilon, possible_action_indices)
            elif exploration_method == 'beam':
                successor_action_ixs = np.argpartition(action_log_probs_v, -num_successors)[-num_successors:]
            else:
                raise ValueError("invalid exploration_method {}".format(exploration_method))

            for successor_number, action_ix in enumerate(reversed(successor_action_ixs)):
                if action_ix == corpus.INVALID_INDEX:
                    action = scone.corpus.INVALID
                else:
                    action = corpus.ACTIONS[action_ix]
                successor_inference_states = [act(inf_state, action)
                                              for act, inf_state in zip(act_fns, inference_states)]
                marginal_log_prob = action_log_probs[int(action_ix)]
                successor_score = beam_item_score + marginal_log_prob

                complete = (action == scone.corpus.INVALID or actions_taken == num_actions_to_take - 1 or len(successor_inference_states[0].possible_actions) == 0)
                successors.append((complete, successor_score, successor_inference_states))

        pruned_successors = sorted(successors, key=lambda t: t[1].npvalue(), reverse=True)[:beam_size]
        new_beam = []
        for (complete, score, inf_states) in pruned_successors:
            if complete:
                completed.append((score, inf_states))
            else:
                new_beam.append((score, inf_states))
        beam = new_beam

    completed = sorted(completed, key=lambda t: t[0].npvalue(), reverse=True)

    # states, actions, score, attention_dists
    if return_beam:
        return [_unpack_beam_item(bi) for bi in completed[:beam_size]]
    else:
        return _unpack_beam_item(completed[0])


class ActionLayer(dy.Saveable):
    def __init__(self,
                 corpus,
                 model,
                 init,
                 input_dim,
                 predict_invalid):
        self.corpus = corpus
        self.input_dim = input_dim
        self.output_dim = len(self.corpus.ACTIONS)
        self._predict_invalid = predict_invalid
        if predict_invalid:
            self.output_dim += 1

    @property
    def predict_invalid(self):
        # backward compatibility
        try:
            return self._predict_invalid
        except:
            self._predict_invalid = False
            return self._predict_invalid

    def get_components(self):
        raise NotImplementedError()

    def restore_components(self, components):
        raise NotImplementedError()

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        # inputs: list of vectors, each of dim self.input_dim
        raise NotImplementedError()

    @property
    def num_attention_heads(self):
        return 1

    @property
    def attention_names(self):
        return ["main"]


class UnfactoredSoftmax(ActionLayer):
    def __init__(self, corpus, model, init, input_dim, predict_invalid):
        super(UnfactoredSoftmax, self).__init__(corpus, model, init, input_dim, predict_invalid)
        self.p_W = model.parameters_from_numpy(init.initialize((self.output_dim, self.input_dim)))

    def get_components(self):
        return (self.p_W, )

    def restore_components(self, components):
        assert len(components) == 1
        self.p_W = components[0]

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        assert len(inputs) == 1
        input = inputs[0]
        support = sorted([self.corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])
        W = dy.parameter(self.p_W)
        return dy.log_softmax(W * input, support)


class Follower(dy.Saveable):
    def __init__(self,
                 model,
                 corpus,
                 vocab_embeddings,
                 embedded_y_dim,
                 enc_state_dim,
                 dec_state_dim,
                 action_layer_constructor=UnfactoredSoftmax,
                 random_seed=None,
                 bidi=False,
                 attention=None,
                 attention_dim=50,
                 dropout=None,
                 ablate_world=False,
                 ablate_language=False,
                 feed_actions_to_decoder=False,
                 constrain_actions_in_loss=True,
                 predict_invalid=False):
        # model: dynet.Model
        # vocab_embeddings: VocabEmbeddings
        self.corpus = corpus
        self.vocab_embeddings = vocab_embeddings

        self.embedded_y_dim = embedded_y_dim

        self.enc_state_dim = enc_state_dim
        self.dec_state_dim = dec_state_dim

        self.z_dim = self.enc_state_dim * (2 if bidi else 1)
        if attention:
            self.z_dim += self.vocab_embeddings.embedding_dim

        init = GlorotInitializer(random_seed)
        self.action_layer = action_layer_constructor(corpus, model, init, self.embedded_y_dim, predict_invalid) # todo: consider projecting down

        self.enc_input_dim = self.vocab_embeddings.embedding_dim
        self.dec_input_dim = (self.z_dim * self.action_layer.num_attention_heads) + self.embedded_y_dim

        if feed_actions_to_decoder:
            self.dec_input_dim += self.corpus.ACTION_DIM

        self.random_seed = random_seed

        self.bidi = bidi
        self.attention = attention
        self.attention_dim = attention_dim

        self.dropout = dropout

        self.ablate_world = ablate_world
        self.ablate_language = ablate_language

        self.feed_actions_to_decoder = feed_actions_to_decoder

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

        #self.p_L0 = model.parameters_from_numpy(init.initialize((len(self.corpus.ACTIONS), self.embedded_y_dim)))
        self.p_E = model.parameters_from_numpy(init.initialize((self.embedded_y_dim, self.corpus.STATE_DIM)))
        self.p_Ls = model.parameters_from_numpy(init.initialize((self.embedded_y_dim, self.dec_state_dim)))
        self.p_Lz = model.parameters_from_numpy(init.initialize((self.embedded_y_dim, self.z_dim)))

        #self.p_b = model.parameters_from_numpy(init.initialize((self.embedded_y_dim)))

        # enc_out_size = self.enc_state_dim * (2 if bidi else 1) * self.num_of_layers
        # dec_hidden_size = self.dec_state_dim * self.num_of_layers
        #
        if attention:
            self.attention_modules = [GlobalAttention(model, init, attention_dim, self.dec_state_dim, self.z_dim)
                                      for _ in range(self.action_layer.num_attention_heads)]

        self._constrain_actions_in_loss = constrain_actions_in_loss
        self._predict_invalid = predict_invalid

    @property
    def constrain_actions_in_loss(self):
        # backward compatibility
        try:
            return self._constrain_actions_in_loss
        except:
            self._constrain_actions_in_loss = True
            return self._constrain_actions_in_loss

    @property
    def predict_invalid(self):
        # backward compatibility
        try:
            return self._predict_invalid
        except:
            self._predict_invalid = False
            return self._predict_invalid

    def actions_to_score(self, state, constrain_actions, predict_invalid):
        if constrain_actions:
            actions =  self.corpus.valid_actions(state)
        else:
            actions =  self.corpus.ACTIONS
        if predict_invalid:
            actions = actions[:]
            actions.append(scone.corpus.INVALID)
        return actions

    def get_components(self):
        components = [self.enc_fwd_lstm, self.dec_lstm, self.action_layer, self.p_E, self.p_Ls, self.p_Lz, self.vocab_embeddings]
        if self.bidi:
            components.append(self.enc_bwd_lstm)
        if self.attention:
            assert len(self.attention_modules) == self.action_layer.num_attention_heads
            components.extend(self.attention_modules)
        return tuple(components)

    def restore_components(self, components):
        k = 7
        self.enc_fwd_lstm, self.dec_lstm, self.action_layer, self.p_E, self.p_Ls, self.p_Lz, self.vocab_embeddings = components[:k]
        if self.bidi:
            self.enc_bwd_lstm = components[k]
            k += 1
        if self.attention:
            self.attention_modules = components[k:k+self.action_layer.num_attention_heads]
            k += self.action_layer.num_attention_heads
        assert  k == len(components)

    def encode_instruction(self, vecs, apply_dropout):
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


    def state_machine(self, testing=True, constrain_actions=True, predict_invalid=False):
        # add params to the computation graph
        # L0 = dy.parameter(self.p_L0)
        E = dy.parameter(self.p_E)
        Ls = dy.parameter(self.p_Ls)
        Lz = dy.parameter(self.p_Lz)
        # b = dy.parameter(self.p_b)


        # TODO: don't stop encoder at utterance boundaries?
        # utterance_tokenwise_encodings = []
        # utterance_summaries = []

        def encode_and_summarize_utterance(utterance):
            xs = self.vocab_embeddings.embed_sequence(utterance)
            # apply dropout if not testing
            zs, summary = self.encode_instruction(xs, not testing)
            utterance_tokenwise_encoding = [dy.concatenate([x, z]) for x, z in zip(xs, zs)]
            return utterance_tokenwise_encoding, summary

        if not testing and self.dropout:
            self.dec_lstm.set_dropout(self.dropout)
        else:
            self.dec_lstm.disable_dropout()

        def initialize_decoder(state):
            return state._replace(
                lstm_state=self.dec_lstm.initial_state(),
                lstm_h=[dy.vecInput(self.dec_state_dim)],
                possible_actions = self.actions_to_score(state.current_world_state, constrain_actions, predict_invalid)
            )

        def observe(state, utterance):
            # update state lstm_state and h with the attended input, and
            # compute the action distribution log_probs for the state
            embedded_state = self.corpus.embed_state(state.current_world_state)
            possible_actions = self.actions_to_score(state.current_world_state, constrain_actions, predict_invalid)

            y = dy.inputVector(embedded_state)
            if self.ablate_world:
                y *= 0

            Ey = E * y

            utterance_tokenwise_encoding, utterance_summary = encode_and_summarize_utterance(utterance)

            if not self.attention:
                #z = dy.average(encoded_utt)
                if self.bidi:
                    # fwd_summary, bwd_summary = utterance_summaries[state.action_count]
                    fwd_summary, bwd_summary = utterance_summary
                    z = dy.concatenate([fwd_summary, bwd_summary])
                else:
                    # fwd_summary, = utterance_summaries[state.action_count]
                    fwd_summary, = utterance_summary
                    z = fwd_summary
                zs = [z for _ in range(self.action_layer.num_attention_heads)]
                attn_dists = None
            else:
                # utt_enc = utterance_tokenwise_encodings[state.action_count]
                # zs, attn_dists = zip(*[attn_mod(state.lstm_h, utt_enc) for attn_mod in self.attention_modules])
                zs, attn_dists = zip(*[attn_mod(state.lstm_h, utterance_tokenwise_encoding)
                                       for attn_mod in self.attention_modules])
                zs = list(zs)
                attn_dists = list(attn_dists)

            if self.ablate_language:
                for i in range(len(zs)):
                    zs[i] *= 0

            dec_inputs = [Ey] + zs
            if self.feed_actions_to_decoder:
                if state.last_action is not None:
                    emb_action = self.corpus.embed_action(state.last_action)
                else:
                    emb_action = np.zeros(self.corpus.ACTION_DIM)
                dec_inputs.append(dy.inputVector(emb_action))

            past_states, past_actions, _ = backchain(state)

            new_lstm_state = state.lstm_state.add_input(dy.concatenate(dec_inputs))
            new_lstm_h = new_lstm_state.h()

            #q = L0 * dy.tanh(Ey + Ls * new_lstm_state.output() + Lz * z + b)
            #q = L0 * (Ey + Ls * new_lstm_state.output() + Lz * z)
            # TODO: consider projecting down
            # TODO: consider also using the embedded last_action here
            qs = [Ey + Ls * new_lstm_state.output() + Lz * z
                  for z in zs]
            # q = Ey + Ls * new_lstm_state.output() + Lz * z

            # support = sorted([self.corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])
            # new_log_probs = dy.log_softmax(q, support)

            new_log_probs = self.action_layer.compute_output_log_probs(qs, possible_actions, state=state.current_world_state, past_states=past_states, past_actions=past_actions)

            return state._replace(lstm_state=new_lstm_state,
                                  lstm_h=new_lstm_h,
                                  log_probs=new_log_probs,
                                  possible_actions=possible_actions,
                                  attention_dist=attn_dists)

        def act(inference_state, action):
            # print(action)
            new_action_count = inference_state.action_count + 1
            assert action in inference_state.possible_actions
            new_state = self.corpus.take_action(inference_state.current_world_state, action)
            if action == scone.corpus.INVALID: # if action was INVALID
                possible_actions = []
            else:
                possible_actions = self.actions_to_score(new_state, constrain_actions, predict_invalid)
            # keep lstm_state and last_h the same (these are updated by observe), but
            # update action-dependent parameters
            return inference_state._replace(
                prev_inference_state=inference_state,
                current_world_state=new_state,
                last_action=action,
                action_count=new_action_count,
                log_probs=None,
                possible_actions=possible_actions,
            )

        return initialize_decoder, observe, act


    def _marginal_loss(self, observe_fn, act_fn, inference_state, action, utterance):
        inference_state = observe_fn(inference_state, utterance)
        assert action in inference_state.possible_actions
        action_ix = self.corpus.ACTIONS_TO_INDEX[action]
        loss = -dy.pick(inference_state.log_probs, action_ix)
        inference_state = act_fn(inference_state, action)
        return loss, inference_state

    def _inf_state_observe_and_act(self, initial_state, testing):
        initialize_decoder, observe, act = self.state_machine(testing=testing, constrain_actions=self.constrain_actions_in_loss, predict_invalid=self.predict_invalid)
        inf_state = initialize_decoder(blank_inference_state(initial_state))
        return inf_state, observe, act

    def loss(self, states, actions, utterances, testing=False, last_utterance_loss=False):
        assert len(states) - 1 == len(actions)
        if not last_utterance_loss:
            assert len(actions) == len(utterances)
        else:
            assert len(utterances) <= len(actions)

        inf_state, observe, act = self._inf_state_observe_and_act(states[0], testing)

        losses = []
        for i in range(len(utterances)):
            loss, inf_state = self._marginal_loss(observe, act, inf_state, actions[i], utterances[i])
            assert inf_state.current_world_state == states[i+1]
            losses.append(loss)

        if last_utterance_loss:
            return losses[-1]
        else:
            return dy.esum(losses)

    def follow(self, ensemble_average_probs, initial_state, utterances, num_actions_to_take=None, sample=False, sample_alpha=1.0, random_state=None, gold_actions=None):
        return ensemble_follow([self], ensemble_average_probs, initial_state, utterances, num_actions_to_take=num_actions_to_take, sample=sample, sample_alpha=sample_alpha, random_state=random_state, gold_actions=gold_actions)


def fmt_state(state):
    if state is not None:
        return ' '.join("%d:%s" % (i+1, contents)
                        for i, contents in enumerate(state))
    else:
        return str(state)


def compare_prediction(agent, instance, predicted_states, predicted_actions, attention_dists):
    def match_print(x, y):
        print("%s\t%s\t%s" % ("+" if x == y else "_", x, y))

    def print_utt(utt):
        print(" \t%s" % ' '.join(s[:6].ljust(6) for s in utt))

    for i in range(len(predicted_actions)):
        print(' '.join(instance.utterances[i]))
        match_print(fmt_state(instance.states[i]), fmt_state(predicted_states[i]))
        match_print(instance.actions[i], predicted_actions[i])
        match_print(fmt_state(instance.states[i+1]), fmt_state(predicted_states[i+1]))
        print_utt(instance.utterances[i])
        if attention_dists:
            for attn_name, attn_dist in zip(agent.action_layer.attention_names, attention_dists[i]):
                print("%s\t%s" % (attn_name, ' '.join("%0.4f" % x for x in attn_dist.npvalue())))
        print()

        pass


def evaluate(instances, pred_statess, pred_actionss, num_actions_to_take=None, name='', print_results=True):
    if num_actions_to_take is None:
        num_actions_to_take = scone.corpus.NUM_TRANSITIONS
    N_instances = len(instances)

    assert N_instances == len(pred_statess) == len(pred_actionss)

    end_state_match_count = 0
    action_match_count = 0
    early_termination_count = 0

    for instance, pred_states, pred_actions in zip(instances, pred_statess, pred_actionss):
        assert len(pred_actions) == len(pred_states) - 1
        if len(pred_actions) < num_actions_to_take:
            early_termination_count += 1

        end_state_matched = pred_states[:num_actions_to_take+1][-1] == instance.states[:num_actions_to_take+1][-1]
        action_matched = pred_actions == instance.actions[:num_actions_to_take]
        if end_state_matched:
            end_state_match_count += 1

        if action_matched:
            assert end_state_matched
            action_match_count += 1

    stats = {
        'acc_end_state': float(end_state_match_count) / N_instances,
        'acc_actions': float(action_match_count) / N_instances,
        'early_termination': float(early_termination_count) / N_instances
    }

    if print_results:
        print("%s terminated early:  %d / %d\t(%0.2f)%%" % (name, early_termination_count, N_instances, 100.0 * early_termination_count / N_instances))
        print("%s end state matches: %d / %d\t(%0.2f)%%" % (name, end_state_match_count, N_instances, 100.0 * end_state_match_count / N_instances))
        print("%s action matches:    %d / %d\t(%0.2f)%%" % (name, action_match_count, N_instances, 100.0 * action_match_count / N_instances))
    return stats

def test_on_instances(agents, instances, beam_size=None, num_actions_to_take=scone.corpus.NUM_TRANSITIONS, verbose=False, name='', utterances_to_follow=None, print_results=True, predict_invalid=False):
    # if utterances_to_follow is not None, use them instead of instance.utterances
    if utterances_to_follow is not None:
        assert len(utterances_to_follow) == len(instances)

    scores = []
    pred_statess = []
    pred_actionss = []
    for ix, instance in enumerate(instances):
        if utterances_to_follow is not None:
            utterances = utterances_to_follow[ix]
            assert len(utterances) == num_actions_to_take
        else:
            utterances = instance.utterances
        dy.renew_cg()
        if beam_size is None:
            pred_states, pred_actions, score, attention_dists = ensemble_follow(agents, False, instance.states[0], utterances, num_actions_to_take=min(num_actions_to_take, len(utterances)), predict_invalid=predict_invalid)
        else:
            pred_states, pred_actions, score, attention_dists = beam_ensemble_follow(
                beam_size, agents, False, instance.states[0], utterances,
                num_actions_to_take=min(num_actions_to_take, len(utterances)),
                return_beam=False, predict_invalid=predict_invalid
            )

        # scores.append(score)
        scores.append(score.npvalue())
        pred_statess.append(pred_states)
        pred_actionss.append(pred_actions)

        if verbose and ix % 20 == 0:
            print(ix)
            print(instance.id)
            compare_prediction(agents[0], instance, pred_states, pred_actions, attention_dists)

    stats = evaluate(instances, pred_statess, pred_actionss,
                   num_actions_to_take=num_actions_to_take, name=name, print_results=print_results)
    if print_results:
        print("%s mean score: %0.4f" % (name, np.mean(scores)))
    return stats

def max_margin_weighting(instance, pred_states, pred_scores_v, args):
    pred_scores_v = np.array(pred_scores_v)
    assert len(pred_states) == len(pred_scores_v)
    # assert len(instance.states) == len(pred_states[0])

    correct_denotations = []
    for i, states in enumerate(pred_states):
        if states[-1] == instance.states[-1]:
            correct_denotations.append(i)

    if not correct_denotations:
        weights = np.zeros_like(pred_scores_v)
    else:
        log_weights = dy.log_softmax(dy.inputVector(pred_scores_v), correct_denotations)
        if args.latent_update_beta != 1.0:
            log_weights *= args.latent_update_beta
            weights = dy.exp(dy.log_softmax(log_weights, correct_denotations)).npvalue()
        else:
            weights = dy.exp(log_weights).npvalue()
        # weights = dy.exp().npvalue()

    return weights, correct_denotations

def train(args):
    if args.embedded_y_dim is None:
        args.embedded_y_dim = args.enc_state_dim

    if args.attention and args.attention_dim is None:
        args.attention_dim = args.enc_state_dim


    if args.save_dir:
        try:
            os.mkdir(args.save_dir)
        except:
            pass

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    corpus, valid_train_instances, valid_dev_instances, test_instances = scone.corpus.load_corpus_and_data(
        # some instances we can't get actions for (train-orig vs train), but if we are training a model with latent
        # actions that's ok
        args, remove_unobserved_action_instances=not args.latent_actions
    )
    print("num actions: {}".format(len(corpus.ACTIONS)))

    if args.corpus == 'alchemy':
        from scone.alchemy_models import AlchemyFactored, AlchemyFactoredSplit, AlchemyFactoredMultihead, AlchemyFactoredMultiheadContextual
        action_layer_constructor_mapping = {
            'unfactored': UnfactoredSoftmax,
            'factored': AlchemyFactored,
            'factored_split': AlchemyFactoredSplit,
            'factored_multihead': AlchemyFactoredMultihead,
            'factored_multihead_contextual': AlchemyFactoredMultiheadContextual,
        }
    elif args.corpus == 'scene':
        from scone.scene_models import SceneFactored, SceneFactoredSplit, SceneFactoredMultihead, SceneFactoredMultiheadContextual
        action_layer_constructor_mapping = {
            'unfactored': UnfactoredSoftmax,
            'factored': SceneFactored,
            'factored_split': SceneFactoredSplit,
            'factored_multihead': SceneFactoredMultihead,
            'factored_multihead_contextual': SceneFactoredMultiheadContextual,
        }
    elif args.corpus == 'tangrams':
        from scone.tangrams_models import TangramsFactored, TangramsFactoredSplit, TangramsFactoredMultihead, TangramsFactoredMultiheadContextual
        action_layer_constructor_mapping = {
            'unfactored': UnfactoredSoftmax,
            'factored': TangramsFactored,
            'factored_split': TangramsFactoredSplit,
            'factored_multihead': TangramsFactoredMultihead,
            'factored_multihead_contextual': TangramsFactoredMultiheadContextual,
        }
    else:
        raise NotImplementedError("no corpus implemented for %s" % args.corpus)

    if args.train_instances_limit:
        valid_train_instances = valid_train_instances[:args.train_instances_limit]

    if args.predict_invalid:
        rng = random.Random(args.random_seed)
        invalid_train_instances = [
            scone.corpus.make_predict_invalid(valid_train_instances, rng)
            for _ in range(int(len(valid_train_instances) * args.invalid_ratio))
        ]
        # reset to get same dev instances if we're filtering train
        rng = random.Random(args.random_seed)
        invalid_dev_instances = [
            scone.corpus.make_predict_invalid(valid_dev_instances, rng)
            for _ in range(int(len(valid_dev_instances) * args.invalid_ratio))
        ]
    else:
        invalid_train_instances = []
        invalid_dev_instances = []

    valid_train_subset_instances = valid_train_instances[:len(valid_train_instances)//10]
    invalid_train_subset_instances = invalid_train_instances[:len(invalid_train_instances)//10]

    train_instances = valid_train_instances + invalid_train_instances
    dev_instances = valid_dev_instances + invalid_dev_instances

    print("{} train instances".format(len(train_instances)))
    print("{} dev instances".format(len(dev_instances)))
    print("{} test instances".format(len(test_instances)))

    model = dy.Model()

    optimizer = dy.AdamTrainer(model)

    if args.load_dir:
        fname = os.path.join(args.load_dir, args.corpus)
        print("loading from %s" % fname)
        follower = model.load(fname)[0]
    else:
        action_layer_constructor = action_layer_constructor_mapping[args.action_layer]

        vocab_counter = Counter([token for instance in valid_train_instances
                                 for utt in instance.utterances
                                 for token in utt])
        vocab = [t for (t, count) in vocab_counter.items() if count > args.unk_threshold]
        vocab_embeddings = OneHotVocabEmbeddings(model, vocab)
        follower = Follower(model,
                            corpus=corpus,
                            vocab_embeddings=vocab_embeddings,
                            embedded_y_dim=args.embedded_y_dim,
                            enc_state_dim=args.enc_state_dim,
                            dec_state_dim=args.dec_state_dim,
                            action_layer_constructor=action_layer_constructor,
                            random_seed=args.random_seed,
                            bidi=args.bidi,
                            attention=args.attention,
                            attention_dim=args.attention_dim,
                            dropout=args.dropout,
                            ablate_world=args.ablate_world,
                            ablate_language=args.ablate_language,
                            feed_actions_to_decoder=args.feed_actions_to_decoder,
                            constrain_actions_in_loss=not args.unconstrained_actions_in_loss,
                            predict_invalid=args.predict_invalid)

    train_stats = []
    dev_stats = []
    test_stats = []

    def save_model():
        if args.save_dir:
            fname = os.path.join(args.save_dir, args.corpus)
            print("saving to %s" % fname)
            model.save(fname, [follower])

    def follow_fn(instances, name, followers=[follower]):
        num_actions_to_take = args.eval_num_actions_to_take or scone.corpus.NUM_TRANSITIONS
        if args.beam_size is not None and args.beam_size > 1:
            print("greedy:")
            test_on_instances(followers, instances, beam_size=None, num_actions_to_take=num_actions_to_take, verbose=args.verbose, name=name+"_greedy")
            print()
            print("beam: ", args.beam_size)
        results = test_on_instances(followers, instances,
                                 beam_size=args.beam_size, num_actions_to_take=num_actions_to_take, verbose=args.verbose, name=name+"_beam")
        print()
        return results

    def loss_fn(instances, name, train=False, latent=False, trace_weighting_function=None):

        if latent:
            assert trace_weighting_function
        if trace_weighting_function:
            assert latent

        losses = []
        if latent:
            num_correct_denotations = []
            num_traces = []
        enum = enumerate(instances)
        if train and not args.hide_progress:
            enum = tqdm.tqdm(enum, desc="instance", total=len(instances), ncols=80)
        for i, instance in enum:
            dy.renew_cg()
            if latent:
                if args.exploration_method in BEAM_EXPLORATION_METHODS:
                    beam = beam_ensemble_follow(
                        args.latent_beam_size, [follower], False, instance.states[0],
                        instance.utterances, num_actions_to_take=None, return_beam=True,
                        predict_invalid=False,
                        # todo: try varying the dropout mask?
                        testing=not train,
                        exploration_method=args.exploration_method,
                    )
                else:
                    raise NotImplementedError('exploration_method {}'.format(args.exploration_method))
                pred_states, pred_actions, pred_scores, _ = zip(*beam)
                pred_scores_v = np.array([score.value() for score in pred_scores])
                weights, correct_denotations = trace_weighting_function(instance, pred_states, pred_scores_v,
                                                                        args=args)
                num_correct_denotations.append(len(correct_denotations))
                num_traces.append(len(beam))
                assert len(weights) == len(pred_scores_v)
                # print(pred_scores)
                # print(weights)
                loss = dy.scalarInput(0.0)
                for score, weight in zip(pred_scores, weights):
                    loss += - score * dy.scalarInput(weight)
                # loss = sum([score * dy.scalarInput(weight) for score, weight in zip(pred_scores, weights)])
            else:
                loss = follower.loss(instance.states, instance.actions, instance.utterances, testing=not train)
            loss_v = loss.value()
            losses.append(loss_v)
            if train:
                loss.backward()
                optimizer.update()
            if latent and args.verbose and i % 1000 == 0:
                mean_correct = np.mean(num_correct_denotations[-1000:])
                mean_traces = np.mean(num_traces[-1000:])
                print_fn = print if args.hide_progress else tqdm.tqdm.write
                print_fn("%s mean correct denotations:\t%0.2f / %d\t(%0.2f)" % (name, mean_correct, mean_traces, mean_correct / mean_traces))
        if train:
            optimizer.status()
        mean_loss = np.mean(losses)
        print("%s loss: %s" % (name, mean_loss))
        if latent:
            mean_correct = np.mean(num_correct_denotations)
            mean_traces = np.mean(num_traces)
            print("%s mean correct denotations:\t%0.2f / %d\t(%0.2f)" % (name, mean_correct, mean_traces, mean_correct / mean_traces))
        print()
        return {'loss': mean_loss}

    def log_train_decode(epochs):
        print("decoding train")
        name = "%d train_subset" % epochs
        stats = {'epoch': epochs}
        stats.update(follow_fn(valid_train_subset_instances, name))
        if not args.latent_actions:
            stats.update(loss_fn(valid_train_subset_instances + invalid_train_subset_instances, name, train=False))
        if args.predict_invalid:
            invalid_name="%d train_subset_invalid" % epochs
            stats.update(follow_fn(valid_train_subset_instances + invalid_train_subset_instances, invalid_name))
        train_stats.append(stats)

    def log_dev_decode(epochs):
        print("decoding dev")
        name = "%d dev" % epochs
        stats = {'epoch': epochs}
        stats.update(follow_fn(valid_dev_instances, name))
        stats.update(loss_fn(dev_instances, name, train=False))
        if args.predict_invalid:
            invalid_name="%d dev_subset_invalid" % epochs
            stats.update(follow_fn(valid_dev_instances + invalid_dev_instances, invalid_name))
        max_dev_acc = max(s['acc_end_state'] for s in dev_stats) if dev_stats else 0
        if stats['acc_end_state'] > max_dev_acc:
            save_model()
            if args.test_decode:
                log_test_decode(epochs)
        dev_stats.append(stats)

    def log_test_decode(epochs):
        print("decoding test")
        name = "%d test" % epochs
        stats = {'epoch': epochs}
        stats.update(follow_fn(test_instances, name))
        stats.update(loss_fn(test_instances, name, train=False))
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
        follower_models = [dy.Model() for _ in args.ensemble_dirs]
        followers = []
        for model, load_dir in zip(follower_models, args.ensemble_dirs):
            fname = os.path.join(load_dir, args.corpus)
            print("loading model from %s" % fname)
            followers.append(model.load(os.path.join(load_dir, args.corpus))[0])

        print("ensemble dev decode:")
        follow_fn(dev_instances, "ensemble dev", followers=followers)

        if args.test_decode:
            print("ensemble test decode:")
            follow_fn(test_instances, "ensemble test", followers=followers)
        sys.exit(0)


    for epoch in range(args.train_epochs):
        print("epoch %d" % epoch)
        random.shuffle(train_instances)
        epoch_start = time.time()
        if args.latent_actions:
            loss_fn(train_instances, "train_latent", train=True, latent=True,
                    trace_weighting_function=max_margin_weighting)
        else:
            loss_fn(train_instances, "train", train=True, latent=False)

        if epoch % args.decode_interval == 0:
            # if epoch % args.decode_interval == 0:
            if not args.latent_actions:
                log_train_decode(epoch)
            log_dev_decode(epoch)
            save_stats()

        # decrease learning rate for sgd
        optimizer.update_epoch()
        print("epoch time: {:.2f} sec".format(time.time() - epoch_start))

    #if (args.train_epochs - 1) % args.decode_interval != 0:
        # log_train_decode(args.train_epochs)
    log_dev_decode(args.train_epochs - 1)
    save_stats()

    return model, follower

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
    parser.add_argument("--attention", action='store_true')
    parser.add_argument("--attention_dim", type=int)
    parser.add_argument("--bidi", action='store_true')
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--random_seed", type=int, default=1)

    parser.add_argument("--unk_threshold", type=int, default=1)  # TODO: up this?

    parser.add_argument("--ablate_world", action='store_true')
    parser.add_argument("--ablate_language", action='store_true')

    parser.add_argument("--test_decode", action='store_true')

    parser.add_argument("--action_layer", choices=[
        "unfactored",
        "factored",
        "factored_split",
        "factored_multihead",
        "factored_multihead_contextual"
    ], default="factored_multihead")

    parser.add_argument("--feed_actions_to_decoder", action='store_true')

    parser.add_argument("--ensemble_dirs", nargs="+")

    parser.add_argument("--unconstrained_actions_in_loss", action='store_true')

    parser.add_argument("--hide_progress", action='store_true')
    parser.add_argument("--predict_invalid", action='store_true')
    parser.add_argument("--invalid_ratio", type=float, default=1.0)

    # latent training
    parser.add_argument("--latent_actions", action='store_true')
    parser.add_argument("--latent_beam_size", type=int)
    parser.add_argument("--exploration_method", default='beam',
                        choices=EXPLORATION_METHODS)
    parser.add_argument("--randomized_beam_epsilon", type=float, default=RANDOMIZED_BEAM_EPSILON)
    parser.add_argument("--latent_update_beta", type=float, default=1.0)

    scone.corpus.add_corpus_args(parser)

    return parser

if __name__ == "__main__":
    util.run(make_parser(), train)
