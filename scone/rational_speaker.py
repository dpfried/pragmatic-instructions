import scone.follower
from scone.follower import Follower
from scone.speaker import Speaker, ensemble_speak, beam_ensemble_speak, fmt_state, in_training_indicator

import os.path
import dynet as dy
import scone.corpus

import util
import random
import numpy as np
import argparse
import pprint
import json

import tqdm

from collections import namedtuple, defaultdict, Counter

SpeakerCandidate = namedtuple("SpeakerCandidate", ["instance", "utterances", "speaker_score", "follower_score"])

SegmentedSpeakerCandidate = namedtuple("SegmentedSpeakerCandidate", ["instance", "action_index", "utterance", "speaker_score", "follower_score", "inference_states"])

def compare_pragmatic_segment(instance, action_ix, segment_utterances, combined_speaker_follower_scores, train_utterances,
                              ref_speaker_score=float('nan'), ref_follower_score=float('nan'), ref_combined_score=float('nan')):
    print("      \t%s" % str(instance.actions[action_ix]))
    print("      \t%s" % fmt_state(instance.states[action_ix]))
    print("%s ref\tcs %+0.3f\tss %+0.3f\tfs %+0.3f\t%s" % (#'*' if cand == best_cand else ' ',
    in_training_indicator(instance.utterances[action_ix], train_utterances),
    ref_combined_score, ref_speaker_score, ref_follower_score,
    ' '.join(instance.utterances[action_ix])))
    print("      \t%s" % fmt_state(instance.states[action_ix+1]))
    for beam_ix, (utt, (combined_score, speaker_score, follower_score)) in \
        enumerate(sorted(zip(segment_utterances, combined_speaker_follower_scores), key=lambda t: t[-1], reverse=True)):
        print("%s %2d\tcs %+0.3f\tss %+0.3f\tfs %+0.3f\t%s" % (#'*' if cand == best_cand else ' ',
            in_training_indicator(utt, train_utterances),
            beam_ix, combined_score, speaker_score, follower_score, ' '.join(utt)))
    print()

def make_score_fn(follower_weight, significant_digits=3):
    def score(candidate):
        # break ties using speaker score, then follower score (with tolerance of 1x10^(-significant_digits) to determine ties)
        fs = round(candidate.follower_score, significant_digits)
        ss = round(candidate.speaker_score, significant_digits)
        combined_score = follower_weight * fs + (1 - follower_weight) * ss
        return (combined_score, ss, fs)
    return score

class RationalSpeaker(object):
    def __init__(self, args, show_progress=False, include_follower_stats=False, max_utterance_length=35):
        self.args = args
        self.follower_models = [dy.Model() for _ in args.follower_dirs]
        self.followers = []
        for model, load_dir in zip(self.follower_models, args.follower_dirs):
            fname = os.path.join(load_dir, args.corpus)
            print("loading model from %s" % fname)
            self.followers.append(model.load(os.path.join(load_dir, args.corpus))[0])

        self.speaker_models = [dy.Model() for _ in args.speaker_dirs]
        self.speakers = []
        for model, load_dir in zip(self.speaker_models, args.speaker_dirs):
            fname = os.path.join(load_dir, args.corpus)
            print("loading model from %s" % fname)
            self.speakers.append(model.load(os.path.join(load_dir, args.corpus))[0])

        self.num_actions_to_take = args.eval_num_actions_to_take or scone.corpus.NUM_TRANSITIONS

        self.max_utterance_length = max_utterance_length

        self.show_progress = show_progress

        self.include_follower_stats = include_follower_stats

    def evaluate(self, instances, candidatess, follower_weight, name='', verbose=False, print_results=True, train_utterances=None):
        pred_utterancess = []
        predictions_by_instance_id = {}
        if verbose:
            print("%s - follower_weight=%s" % (name, follower_weight))
            comparison_stats = Counter()
        else:
            comparison_stats = None
        it = enumerate(zip(instances, candidatess))
        if self.show_progress and not verbose:
            it = tqdm.tqdm(it, total=len(instances), desc=name)
        for ix, (instance, candidates) in it:
            this_verbose = verbose and ix % 5 == 0
            if this_verbose:
                print(ix)
            pred_utts = self.predict_single(instance, candidates, follower_weight, verbose=this_verbose,
                                            train_utterances=train_utterances, comparison_stats=comparison_stats)
            pred_utterancess.append(pred_utts)
            assert instance.id not in predictions_by_instance_id
            predictions_by_instance_id[instance.id] = pred_utts
        eval_results = scone.speaker.evaluate(instances, pred_utterancess, num_actions_to_take=self.num_actions_to_take,
                                              name=name, print_results=print_results)

        if self.include_follower_stats:
            # get completion accuracy etc.
            follower_stats = scone.follower.test_on_instances(self.followers, instances, beam_size=None, utterances_to_follow=pred_utterancess, print_results=False)
            # insert these into eval_results
            eval_results = dict(eval_results, **follower_stats)

        if verbose:
            print("%s: reference better than predicted in speaker score:\t%d / %d" % (name, comparison_stats['ref_speaker_higher'], comparison_stats['compared']))
            print("%s: reference better than predicted in follower score:\t%d / %d" % (name, comparison_stats['ref_follower_higher'], comparison_stats['compared']))
        return eval_results, predictions_by_instance_id

    def _follower_score(self, candidate):
        states = candidate.instance.states[:self.num_actions_to_take+1]
        actions = candidate.instance.actions[:self.num_actions_to_take]
        utterances = candidate.utterances
        def follower_score(follower):
            dy.renew_cg()
            return -1 * follower.loss(states, actions, utterances, testing=True).npvalue()

        follower_scores = [
            follower_score(follower) for follower in self.followers
        ]
        return np.mean(follower_scores)

    def get_candidates(self, instance):
        states = instance.states[:self.num_actions_to_take+1]
        actions = instance.actions[:self.num_actions_to_take]
        if self.args.inference_type == "sample":
            candidates = []
            if self.args.force_length_match:
                length_constraints = [len(utt) for utt in instance.utterances]
            else:
                length_constraints = None
            for i in range(self.args.num_candidates):
                dy.renew_cg()
                utterances, scores = zip(*ensemble_speak(self.speakers, False, states, actions,
                                                         sample=True, sample_alpha=self.args.sample_alpha,
                                                         max_utterance_length=self.max_utterance_length, length_constraints=length_constraints))
                assert len(utterances) == self.num_actions_to_take
                speaker_score = np.sum(scores)
                if any(not u for u in utterances):
                    continue
                candidate = SpeakerCandidate(instance=instance, utterances=utterances,
                                             speaker_score=speaker_score, follower_score=None)
                candidate = candidate._replace(follower_score=self._follower_score(candidate))
                candidates.append(candidate)
        else:
            raise NotImplementedError()
        return candidates

    def predict_single(self, instance, candidates, follower_weight, verbose=False, train_utterances=None):
        assert all(cand.instance == instance for cand in candidates)
        score_fn = make_score_fn(follower_weight)
        best_cand = max(candidates, key=score_fn)

        if verbose:
            print(instance)
            combined_speaker_follower_scores = [score_fn(cand) for cand in candidates]

            for action_ix in range(self.num_actions_to_take):
                segment_utts = [cand.utterances[action_ix] for cand in candidates]
                compare_pragmatic_segment(instance, action_ix, segment_utts, combined_speaker_follower_scores, train_utterances)

        return best_cand.utterances


class RationalSpeakerSentenceSegmented(RationalSpeaker):
    def get_candidates(self, instance):
        states = instance.states[:self.num_actions_to_take+1]
        actions = instance.actions[:self.num_actions_to_take]
        cands_by_action_index = defaultdict(list)
        if self.args.force_length_match:
            length_constraints = [len(utt) for utt in instance.utterances]
        else:
            length_constraints = None
        if self.args.inference_type == "sample":
            for i in range(self.args.num_candidates):
                dy.renew_cg()
                utterances, scores = zip(*ensemble_speak(self.speakers, False, states, actions,
                                                         sample=True, sample_alpha=self.args.sample_alpha,
                                                         max_utterance_length=self.max_utterance_length, length_constraints=length_constraints))
                assert len(utterances) == self.num_actions_to_take
                assert len(scores) == self.num_actions_to_take
                for action_ix, (utt, score) in enumerate(zip(utterances, scores)):
                    if not utt:
                        continue
                    cands_by_action_index[action_ix].append(SegmentedSpeakerCandidate(instance=instance, action_index=action_ix,
                                                                                      utterance=utt, speaker_score=score, follower_score=None, inference_states=None))
        elif self.args.inference_type == "beam":
            dy.renew_cg()
            utterance_beams = beam_ensemble_speak(self.args.num_candidates, self.speakers, False, states, actions,
                                                  return_beam=True, max_utterance_length=self.max_utterance_length, length_constraints=length_constraints)
            for action_ix, beam in enumerate(utterance_beams):
                cands_by_action_index[action_ix] = [
                    SegmentedSpeakerCandidate(instance=instance, action_index=action_ix,
                                              utterance=utt, speaker_score=score, follower_score=None, inference_states=None)
                    for (utt, score) in beam
                    if len(utt) > 0
                ]
        else:
            raise NotImplementedError()
        return cands_by_action_index


    def predict_single(self, instance, cands_by_action_index, follower_weight, verbose=False, train_utterances=None, comparison_stats=None):
        states = instance.states[:self.num_actions_to_take+1]
        actions = instance.actions[:self.num_actions_to_take]
        ref_utterances = instance.utterances[:self.num_actions_to_take]

        if verbose:
            def ref_speaker_scores(speaker):
                dy.renew_cg()
                return [-1 * l.npvalue()
                        for l in speaker.loss(states, actions, ref_utterances, testing=True, loss_per_action=True)]
            # average reference-wise scores across speakers
            ref_speaker_scores = np.array([ref_speaker_scores(speaker) for speaker in self.speakers]).mean(axis=0)

        pred_utterances = []

        dy.renew_cg()
        inf_states, observe_fns, act_fns = zip(*[follower._inf_state_observe_and_act(states[0], testing=True)
                                               for follower in self.followers])
        score_fn = make_score_fn(follower_weight)

        # def follower_score(follower, all_utterances):
        #     dy.renew_cg()
        #     return -1 * follower.loss(states, actions, all_utterances, testing=True, last_utterance_loss=True).npvalue()

        for action_ix in range(self.num_actions_to_take):
            rescored_candidates = []
            for candidate in cands_by_action_index[action_ix]:
                assert candidate.instance == instance
                # all_utterances = pred_utterances + [candidate.utterance]
                marginal_ret = [
                    follower._marginal_loss(observe, act, inf_state, actions[action_ix], candidate.utterance)
                    for follower, inf_state, observe, act in zip(self.followers, inf_states, observe_fns, act_fns)
                ]
                follower_scores = [-1 * l.npvalue() for (l, inf_state) in marginal_ret]
                new_inf_states = [inf_state for (l, inf_state) in marginal_ret]
                rescored_candidates.append(candidate._replace(follower_score=np.mean(follower_scores),
                                                              inference_states=new_inf_states))

            best_cand = max(rescored_candidates, key=score_fn)

            if verbose:
                ref_speaker_score = ref_speaker_scores[action_ix]
                ref_follower_score = np.mean([
                    -1 * follower._marginal_loss(observe, act, inf_state, actions[action_ix], ref_utterances[action_ix])[0].npvalue()
                    for follower, inf_state, observe, act in zip(self.followers, inf_states, observe_fns, act_fns)
                    ])
                ref_combined_score = follower_weight * ref_follower_score + (1 - follower_weight) * ref_speaker_score

                for cand in rescored_candidates:
                    if ref_utterances[action_ix] == cand.utterance:
                        assert util.close(ref_speaker_score, cand.speaker_score)
                        assert util.close(ref_follower_score, cand.follower_score)

                print(instance)
                print("\t".join(" ".join(utt) for utt in pred_utterances))
                combined_speaker_follower_scores = [score_fn(cand) for cand in rescored_candidates]
                segment_utts = [cand.utterance for cand in rescored_candidates]
                compare_pragmatic_segment(instance, action_ix, segment_utts, combined_speaker_follower_scores, train_utterances,
                                          ref_speaker_score=ref_speaker_score, ref_follower_score=ref_follower_score, ref_combined_score=ref_combined_score)

                if comparison_stats is not None:
                    if ref_speaker_score > best_cand.speaker_score:
                        comparison_stats['ref_speaker_higher'] += 1
                    if ref_follower_score > best_cand.follower_score:
                        comparison_stats['ref_follower_higher'] += 1
                    comparison_stats['compared'] += 1

            pred_utterances.append(best_cand.utterance)
            inf_states = best_cand.inference_states


        return pred_utterances

def run(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    corpus, train_instances, dev_instances, test_instances = scone.corpus.load_corpus_and_data(args)

    include_follower_stats = any(metric in ["acc_end_state", "acc_actions"]
                                 for metric in args.tuning_metrics)
    max_utterance_length = max(len(utt) for instance in train_instances + dev_instances
                               for utt in instance.utterances) + 3

    if args.sentence_segmented:
        rational_speaker = RationalSpeakerSentenceSegmented(args, show_progress=True, include_follower_stats=include_follower_stats, max_utterance_length=max_utterance_length)
    else:
        rational_speaker = RationalSpeaker(args, show_progress=False, include_follower_stats=include_follower_stats, max_utterance_length=max_utterance_length)

    dev_candidatess = [rational_speaker.get_candidates(instance)
                       for instance in tqdm.tqdm(dev_instances, desc='dev-cands')]

    dev_results_by_weight = {
        follower_weight: rational_speaker.evaluate(dev_instances, dev_candidatess, follower_weight, name="dev-%0.2f" % follower_weight, verbose=False, print_results=False)
        for follower_weight in np.arange(0, args.granularity + 1) / float(args.granularity)
    }

    # dev_bleus_by_weight = {
    #     0: 0,
    #     1: 1
    # }

    best_weights_by_metric = {
        metric: max(dev_results_by_weight.items(), key=lambda t: t[1][0][metric])[0]
        for metric in args.tuning_metrics
    }

    # dev_s0_results, dev_s0_preds = dev_results_by_weight[0.0]
    # dev_l0_results, dev_l0_preds = dev_results_by_weight[1.0]
    # dev_s0_l0_results, dev_s0_l0_preds = dev_results_by_weight[best_weight]

    def save_predictions(partition_name, results_by_weight):
        if args.prediction_output_dir:
            output_path = os.path.join(args.prediction_output_dir, "%s.%s.predictions.json" % (args.corpus, partition_name))
            preds = [
                {
                    'follower_weight': follower_weight,
                    'metrics': metrics,
                    'predictions': predictions
                } for follower_weight, (metrics, predictions) in results_by_weight.items()
            ]
            with open(output_path, 'w') as f:
                json.dump(preds, f)

    def print_stats(partition_name, results_by_weight):
        print("%s stats:" % partition_name)
        print(json.dumps(
            list((k, s) for k, (s, p) in sorted(results_by_weight.items())),
            indent=2
        ))

    save_predictions("dev", dev_results_by_weight)

    train_utterances = set(tuple(utt) for inst in train_instances
                           for utt in inst.utterances)

    if args.verbose:
        rational_speaker.evaluate(dev_instances, dev_candidatess, 0.0, name='dev-S0', verbose=True, train_utterances=train_utterances)
        rational_speaker.evaluate(dev_instances, dev_candidatess, 1.0, name='dev-L0', verbose=True, train_utterances=train_utterances)
        for metric, weight in  best_weights_by_metric.items():
            rational_speaker.evaluate(dev_instances, dev_candidatess, weight, name='dev-S0+L0(%s)' % metric, verbose=True, train_utterances=train_utterances)

    print_stats("dev", dev_results_by_weight)

    for metric, weight in best_weights_by_metric.items():
        print("best dev follower_weight (%s):\t%s" % (metric, weight))
        print("best dev stats (%s):\t%s" % (metric, dev_results_by_weight[weight][0]))

    if args.test_decode:
        test_candidatess = [rational_speaker.get_candidates(instance)
                            for instance in tqdm.tqdm(test_instances, desc="test-cands")]
        weights_and_names = [(0.0, "test-S0"), (1.0, "test-L0")]
        for metric, weight in best_weights_by_metric.items():
            weights_and_names.append((weight, "test-S0+L0(%s)" % metric))
        test_results_by_weight = {
            follower_weight: rational_speaker.evaluate(test_instances, test_candidatess, follower_weight, name=name, verbose=args.verbose, train_utterances=train_instances)
            for follower_weight, name in weights_and_names
        }
        print_stats("test", test_results_by_weight)
        save_predictions("test", test_results_by_weight)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--follower_dirs", nargs="+")
    parser.add_argument("--speaker_dirs", nargs="+")
    parser.add_argument("--verbose", action='store_true')

    parser.add_argument("--random_seed", type=int, default=1)

    parser.add_argument("--inference_type", choices=["beam", "sample"])
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--sample_alpha", type=float, default=1.0)

    parser.add_argument("--eval_num_actions_to_take", type=int)

    parser.add_argument("--test_decode", action='store_true')

    parser.add_argument("--force_length_match", action='store_true')

    parser.add_argument("--sentence_segmented", action='store_true')

    parser.add_argument("--granularity", type=int, default=20)

    parser.add_argument("--prediction_output_dir")

    parser.add_argument("--tuning_metrics", nargs="+",
                        choices=["bleu", "unpenalized_bleu", "acc_end_state", "acc_actions"],
                        default=["bleu", "acc_end_state"])

    scone.corpus.add_corpus_args(parser)

    return parser

if __name__ == "__main__":
    util.run(make_parser(), run)
