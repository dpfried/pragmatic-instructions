from collections import namedtuple
from os.path import join
from os.path import expanduser

from argparse import ArgumentParser

Instance = namedtuple("Instance", "id, states, actions, utterances")

NUM_TRANSITIONS = 5

def add_corpus_args(parser: ArgumentParser):
    group = parser.add_argument_group('data')
    group.add_argument("--corpus", choices=['alchemy', 'scene', 'tangrams'])
    group.add_argument("--corpora_dir", default='data/scone/rlong')
    group.add_argument("--train_original", action='store_true')
    group.add_argument("--train_original_no_add_actions", action='store_true')
    group.add_argument("--train_original_length_1", action='store_true')
    group.add_argument("--max_transitions_per_train_instance", type=int, default=NUM_TRANSITIONS)

def load_corpus_and_data(args, remove_unobserved_action_instances=True):
    if args.corpus == 'alchemy':
        from scone.alchemy import AlchemyCorpus
        corpus = AlchemyCorpus()
    elif args.corpus == 'scene':
        from scone.scene import SceneCorpus
        corpus = SceneCorpus()
    elif args.corpus == 'tangrams':
        from scone.tangrams import TangramsCorpus
        corpus = TangramsCorpus()
    else:
        raise NotImplementedError("no corpus implemented for %s" % args.corpus)

    train_instances, dev_instances, test_instances = corpus.load_train_dev_test(
        args.corpora_dir, train_original=args.train_original, train_original_no_add_actions=args.train_original_no_add_actions,
        remove_unobserved_action_instances=remove_unobserved_action_instances,
        max_transitions_per_train_instance=args.max_transitions_per_train_instance,
    )

    return corpus, train_instances, dev_instances, test_instances

INVALID = "Invalid"

def make_predict_invalid(instances, random_state):
    instance_to_corrupt = random_state.choice(instances)
    instance_to_copy = random_state.choice(instances)
    index_to_replace = random_state.randint(0, len(instance_to_corrupt.actions) - 1)

    new_instance = Instance(
        id=instance_to_corrupt.id+"-corrupt",
        states=instance_to_corrupt.states[:index_to_replace+1] + [None],
        actions=instance_to_corrupt.actions[:index_to_replace] + [INVALID],
        utterances=instance_to_corrupt.utterances[:index_to_replace] + [instance_to_copy.utterances[index_to_replace]],
    )

    return new_instance


def try_align(inst, full_inst):
    sub_utterance_length = len(inst.utterances)
    assert sub_utterance_length == len(inst.states) - 1

    def matches(start_position):
        matches_start = inst.states[0] == full_inst.states[start_position]
        matches_end = inst.states[sub_utterance_length] == full_inst.states[start_position+sub_utterance_length]
        return matches_start and matches_end

    start_positions = [i for i in range(len(full_inst.utterances) - sub_utterance_length + 1)
                       if full_inst.utterances[i:i+sub_utterance_length] == inst.utterances
                       if matches(i)]

    if not start_positions:
        return None

    if len(start_positions) > 1:
        print("for sub instance {} and full instance {} there are multiple match positions; picking the first".format(inst.id, full_inst.id))

    start = start_positions[0]

    utterances = full_inst.utterances[start:start+len(inst.utterances)]
    assert utterances == inst.utterances
    actions = full_inst.actions[start:start+len(inst.utterances)]
    states = full_inst.states[start:start+len(inst.utterances)+1]
    if inst.actions is not None:
        assert inst.actions == actions
        assert inst.states == states
    assert len(inst.states) == len(states) and inst.states[0] == states[0] and inst.states[-1] == states[-1]
    assert len(actions) == len(inst.utterances)
    return inst._replace(actions=actions, states=states)

class Corpus(object):

    ACTIONS = None

    STATE_DIM = None

    ACTION_DIM = None

    ACTION_IN_STATE_CONTEXT_DIM = None

    @property
    def ACTIONS_TO_INDEX(self):
        try:
            return self._ACTIONS_TO_INDEX
        except:
            self._ACTIONS_TO_INDEX = {
                action: index
                for (index, action) in enumerate(self.ACTIONS)
            }
            self._ACTIONS_TO_INDEX[INVALID] = len(self._ACTIONS_TO_INDEX)
            return self._ACTIONS_TO_INDEX

    @property
    def INVALID_INDEX(self):
        return self.ACTIONS_TO_INDEX[INVALID]

    def dataset_name(self):
        raise NotImplementedError()


    def parse_utterance(self, utterance_string):
        return utterance_string.split()


    def parse_state(self, state_string):
        raise NotImplementedError()


    def parse_instance_line(self, line, max_transitions):
        sections = line.split('\t')
        num_sections = len(sections)
        assert num_sections % 2 == 0
        num_transitions = (num_sections - 2) // 2
        #assert len(sections) == NUM_TRANSITIONS * 2 + 2

        id_ = sections[0]

        states = []
        utterances = []

        latent_state = False

        for i in range(1, len(sections), 2):
            state_str = sections[i]
            if state_str == '?':
                states.append(None)
                latent_state = True
            else:
                states.append(self.parse_state(state_str))

            if i < len(sections) - 1:
                utterance_str = sections[i+1]
                utterances.append(self.parse_utterance(utterance_str))

        assert len(states) == num_transitions + 1
        assert len(utterances) == num_transitions

        if not latent_state:
            actions = [
                self.action_taken(s1, s2)
                for s1, s2 in zip(states, states[1:])
            ]
            assert len(actions) == num_transitions
        else:
            actions = None

        states = states[:max_transitions+1]
        actions = actions[:max_transitions]
        utterances = utterances[:max_transitions]

        return Instance(id=id_, states=states, actions=actions, utterances=utterances)


    def load_data(self, filename, max_transitions):
        with open(filename) as f:
            instances = [
                self.parse_instance_line(line, max_transitions) for line in f
            ]
        return instances


    def load_fold(self, root_dir, fold, max_transitions):
        fname = join(root_dir, "%s-%s.tsv" % (self.dataset_name(), fold))
        return self.load_data(fname, max_transitions)


    def load_train_dev_test(self, root_dir, train_original=False, train_original_no_add_actions=False,
                            remove_unobserved_action_instances=True, max_transitions_per_train_instance=NUM_TRANSITIONS):
        train_name = "train-orig" if train_original else "train"
        train_data = self.load_fold(root_dir, train_name, max_transitions_per_train_instance)
        dev_data = self.load_fold(root_dir, "dev", NUM_TRANSITIONS)
        test_data = self.load_fold(root_dir, "test", NUM_TRANSITIONS)
        if train_original and not train_original_no_add_actions:
            train_full_by_id = {
                inst.id: inst for inst in self.load_fold(root_dir, "train", max_transitions_per_train_instance)
            }
            def update(inst):
                keys = ["train-" + pfx + inst.id for pfx in ["", "A", "B", "C", "D", "E"]]
                # there's a many-to-one mapping for alchemy so we'll check for matching utterances
                for key in keys:
                    if key not in train_full_by_id:
                        continue
                    full_inst = train_full_by_id[key]
                    maybe_aligned = try_align(inst, full_inst)
                    if maybe_aligned is not None:
                        return maybe_aligned
                print("couldn't find full instance for id: ", inst.id)
                return inst
            train_data = [update(inst) for inst in train_data]

        if remove_unobserved_action_instances:
            print("removing unobserved action instances")
            print("train data before filtering: {}".format(len(train_data)))
            train_data = [inst for inst in train_data if inst.actions is not None]
            print("train data after filtering: {}".format(len(train_data)))
            dev_data_filtered = [inst for inst in dev_data if inst.actions is not None]
            test_data_filtered = [inst for inst in test_data if inst.actions is not None]
            assert len(dev_data) == len(dev_data_filtered)
            assert len(test_data) == len(test_data_filtered)
            dev_data = dev_data_filtered
            test_data = test_data_filtered

        return train_data, dev_data, test_data


    def action_taken(self, initial_state, next_state):
        valid_actions = self.valid_actions(initial_state)
        for action in valid_actions:
            successor = self.take_action(initial_state, action)
            if successor == next_state:
                return action
        raise ValueError("no action in valid actions reaches next state")


    def valid_actions(self, state):
        raise NotImplementedError()

    def take_action(self, state, action):
        raise NotImplementedError()

    def embed_state(self, state):
        raise NotImplementedError()

    def embed_action(self, action):
        raise NotImplementedError()

    def embed_action_in_state_context(self, action, state_before, state_after, past_states, past_actions):
        raise NotImplementedError()

