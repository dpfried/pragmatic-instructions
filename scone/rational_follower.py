import scone.follower
from scone.follower import Follower, ensemble_follow, beam_ensemble_follow
from scone.speaker import Speaker

import os.path
import dynet as dy

import util
import random
import numpy as np
import argparse
import pprint

import tqdm

from collections import namedtuple
import scone.corpus
import json

FollowerCandidate = namedtuple("FollowerCandidate", ["instance", "states", "actions", "follower_score", "speaker_score"])

def run(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    corpus, train_instances, dev_instances, test_instances = scone.corpus.load_corpus_and_data(args)

    num_actions_to_take = args.eval_num_actions_to_take or scone.corpus.NUM_TRANSITIONS

    follower_models = [dy.Model() for _ in args.follower_dirs]
    followers = []
    for model, load_dir in zip(follower_models, args.follower_dirs):
        fname = os.path.join(load_dir, args.corpus)
        print("loading model from %s" % fname)
        followers.append(model.load(os.path.join(load_dir, args.corpus))[0])

    if not args.speaker_dirs:
        print("warning: no speakers are being loaded, will just use follower score")
        args.speaker_dirs = []
    speaker_models = [dy.Model() for _ in args.speaker_dirs]
    speakers = []
    for model, load_dir in zip(speaker_models, args.speaker_dirs):
        fname = os.path.join(load_dir, args.corpus)
        print("loading model from %s" % fname)
        speakers.append(model.load(os.path.join(load_dir, args.corpus))[0])

    def get_candidates(instance):
        if args.inference_type == "sample":
            candidates = []
            for i in range(args.num_candidates):
                dy.renew_cg()
                states, actions, follower_score, _ = \
                    ensemble_follow(followers, False, instance.states[0], instance.utterances,
                                    num_actions_to_take=num_actions_to_take, sample=True, sample_alpha=args.sample_alpha)
                follower_score_v = follower_score.npvalue()
                if len(actions) != num_actions_to_take:
                    continue
                candidates.append(FollowerCandidate(instance=instance, states=states, actions=actions, follower_score=follower_score_v, speaker_score=None))
        else:
            dy.renew_cg()
            beam = beam_ensemble_follow(args.num_candidates, followers, False, instance.states[0], instance.utterances,
                                        num_actions_to_take=num_actions_to_take, return_beam=True)
            candidates = [FollowerCandidate(instance=instance, states=states, actions=actions, follower_score=follower_score.npvalue(), speaker_score=None)
                       for (states, actions, follower_score, _) in beam
                          if len(actions) == num_actions_to_take]
        return candidates

    def speaker_score_candidate(candidate):
        if not speakers:
            return 0.0
        def speaker_score(speaker):
            dy.renew_cg()
            return -1 * speaker.loss(candidate.states, candidate.actions, candidate.instance.utterances, testing=True).npvalue()

        speaker_scores = [
            speaker_score(speaker) for speaker in speakers
        ]
        return np.mean(speaker_scores)

    def add_speaker_scores(candidates):
        return [cand._replace(speaker_score=speaker_score_candidate(cand)) for cand in candidates]

    def evaluate(candidatess, speaker_weight, name='', verbose=False, print_results=True):
        pred_statess = []
        pred_actionss = []
        instances = []
        if verbose:
            print("%s - speaker_weight=%s" % (name, speaker_weight))

        predictions_by_instance_id = {}

        for ix, candidates in enumerate(candidatess):
            instance = candidates[0].instance
            assert all(cand.instance == instance for cand in candidates[1:])
            instances.append(instance)
            best_cand = max(candidates, key=lambda cand: speaker_weight * cand.speaker_score + (1 - speaker_weight) * cand.follower_score)
            pred_statess.append(best_cand.states)
            pred_actionss.append(best_cand.actions)

            if verbose and ix % 5 == 0:
                print(ix)
                print(instance)
                scone.follower.compare_prediction(followers[0], candidates[0].instance,
                                                  best_cand.states, best_cand.actions, None)
            predictions_by_instance_id[instance.id] = (best_cand.actions, best_cand.states)
        eval_results = scone.follower.evaluate(instances, pred_statess, pred_actionss,
                                       num_actions_to_take=num_actions_to_take, print_results=print_results,
                                       name=name)
        return eval_results, predictions_by_instance_id

    dev_candidatess = [add_speaker_scores(get_candidates(instance))
                       for instance in tqdm.tqdm(dev_instances)]

    dev_results_by_speaker_weight = {
        speaker_weight: evaluate(dev_candidatess, speaker_weight, verbose=False, print_results=False)
        for speaker_weight in np.arange(0, 20 + 1) / 20.0
    }

    dev_accuracies_by_speaker_weight = {
        weight: metrics['acc_end_state']
        for weight, (metrics, predictions) in dev_results_by_speaker_weight.items()
    }

    best_speaker_weight = max(dev_accuracies_by_speaker_weight.items(), key=lambda t: t[1])[0]

    def save_predictions(partition_name, results_by_weight):
        if args.prediction_output_dir:
            output_path = os.path.join(args.prediction_output_dir, "%s.%s.predictions.json" % (args.corpus, partition_name))
            preds = [
                {
                    'speaker_weight': speaker_weight,
                    'accuracy': accuracy,
                    'predictions': predictions
                } for speaker_weight, (accuracy, predictions) in results_by_weight.items()
            ]
            with open(output_path, 'w') as f:
                json.dump(preds, f)

    save_predictions("dev", dev_results_by_speaker_weight)

    if args.verbose:
        evaluate(dev_candidatess, 0.0, name='dev-L0', verbose=True)
        evaluate(dev_candidatess, 1.0, name='dev-S0', verbose=True)
        evaluate(dev_candidatess, best_speaker_weight, name='dev-L0+S0', verbose=True)

    print("dev accuracies:")
    print(list(sorted(dev_accuracies_by_speaker_weight.items())))

    print("best dev speaker_weight:\t%s" % best_speaker_weight)
    print("best dev accuracy:\t%s" % dev_accuracies_by_speaker_weight[best_speaker_weight])

    if args.test_decode:
        test_candidatess = [add_speaker_scores(get_candidates(instance))
                            for instance in tqdm.tqdm(test_instances)]

        test_results_by_speaker_weight = {}

        weights_and_names = [(0.0, "test-L0"), (1.0, "test-S0"), (best_speaker_weight, "test-L0+S0")]
        for weight, name in weights_and_names:
            test_results_by_speaker_weight[weight] = evaluate(test_candidatess, weight, name=name, verbose=args.verbose)

        save_predictions("test", test_results_by_speaker_weight)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--follower_dirs", nargs="+")
    parser.add_argument("--speaker_dirs", nargs="*")
    parser.add_argument("--verbose", action='store_true')

    parser.add_argument("--random_seed", type=int, default=1)

    parser.add_argument("--inference_type", choices=["beam", "sample"])
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--sample_alpha", type=float, default=1.0)

    parser.add_argument("--eval_num_actions_to_take", type=int)

    parser.add_argument("--test_decode", action='store_true')

    parser.add_argument("--prediction_output_dir")

    scone.corpus.add_corpus_args(parser)

    return parser

if __name__ == "__main__":
    util.run(make_parser(), run)
