from .corpus import Corpus, INVALID, NUM_TRANSITIONS
from collections import namedtuple
import numpy as np

Insert = namedtuple("Insert", ["shape", "ix", "t"])

def insert(shape, ix):
    return Insert(shape=shape, ix=ix, t="tan_i")

Remove = namedtuple("Remove", ["ix", "t"])

def remove(ix):
    return Remove(ix=ix, t="tan_r")

Swap = namedtuple("Swap", ["first_ix", "second_ix", "t"])

def swap(first_ix, second_ix):
    return Swap(first_ix=first_ix, second_ix=second_ix, t="tan_s")

MAX_POSITIONS = 5

SHAPES = set(range(MAX_POSITIONS))

NUM_SHAPES = len(SHAPES)

class TangramsCorpus(Corpus):

    ACTIONS = \
        [
            insert(shape, ix)
            for shape in range(MAX_POSITIONS)
            for ix in range(MAX_POSITIONS) # can only insert if we have < 5 shapes
        ] + [
            remove(ix)
            for ix in range(MAX_POSITIONS)
        ] + [
            swap(first_ix, second_ix)
            for first_ix in range(MAX_POSITIONS)
            for second_ix in range(first_ix+1, MAX_POSITIONS)
        ]


    def dataset_name(self):
        return "tangrams"


    def parse_token(self, token):
        ix, content = token.split(":")
        shape = int(content)
        assert shape in SHAPES
        return shape


    def parse_state(self, state_string):
        toks = state_string.split()
        assert len(toks) <= MAX_POSITIONS
        return [self.parse_token(token) for token in toks]


    def valid_actions(self, state):
        def it():
            num_shapes_present = len(state)
            assert not (set(state) - SHAPES)
            missing_shapes = SHAPES - set(state)
            for shape in missing_shapes:
                for ix in range(num_shapes_present + 1):
                    assert ix <= MAX_POSITIONS
                    yield insert(shape, ix)
            for ix in range(num_shapes_present):
                yield remove(ix)
                for other_ix in range(ix+1, num_shapes_present):
                    if other_ix == ix:
                        continue
                    yield swap(ix, other_ix)
        return list(it())


    def take_action(self, state, action):
        new_state = state[:]

        if isinstance(action, Insert):
            assert 0 <= action.ix <= len(new_state)
            assert action.shape not in new_state
            new_state.insert(action.ix, action.shape)
        elif isinstance(action, Remove):
            assert 0 <= action.ix < len(new_state)
            new_state.pop(action.ix)
        elif isinstance(action, Swap):
            L = len(new_state)
            assert 0 <= action.first_ix < L
            assert 0 <= action.second_ix < L
            new_state[action.first_ix], new_state[action.second_ix] = new_state[action.second_ix], new_state[action.first_ix]
        elif action == INVALID:
            return None
        else:
            raise ValueError("bad action type", action)
        return new_state

    STATE_DIM = MAX_POSITIONS * NUM_SHAPES


    def _embed_state_notflat(self, state):
        assert len(state) <= MAX_POSITIONS

        x = np.zeros((MAX_POSITIONS, NUM_SHAPES))

        for ix, shape in enumerate(state):
            x[ix,shape] = 1
        return x


    def embed_state(self, state):
        return self._embed_state_notflat(state).flatten()


    ACTION_DIM = 3 + MAX_POSITIONS + 2 * NUM_SHAPES

    def embed_action(self, action):
        action_type = np.zeros(3)
        first_ix = np.zeros(MAX_POSITIONS)
        second_ix = np.zeros(MAX_POSITIONS)
        shape = np.zeros(NUM_SHAPES)
        if isinstance(action, Insert):
            action_type[0] = 1
            first_ix[action.ix] = 1
            shape[action.shape] = 1
        elif isinstance(action, Remove):
            action_type[1] = 1
            first_ix[action.ix] = 1
        elif isinstance(action, Swap):
            action_type[2] = 1
            first_ix[action.first_ix] = 1
            second_ix[action.second_ix] = 1
        elif action == INVALID:
            return None
        else:
            raise ValueError("bad action type", action)
        return np.hstack([action_type, first_ix, second_ix, shape])


    # ACTION_IN_STATE_CONTEXT_DIM = NUM_TRANSITIONS * NUM_SHAPES
    # def embed_action_in_state_context(self, action, state_before, state_after, past_states, past_actions):
    #     x = np.zeros((NUM_TRANSITIONS, NUM_SHAPES))
    #
    #     for action_ix, (action, state) in enumerate(zip(past_actions, past_states)):
    #         if isinstance(action, Remove):
    #             removed_ix = action.ix
    #             removed_shape = state[removed_ix]
    #             assert removed_shape in SHAPES
    #             if action_ix < len(past_states) - 1:
    #                 assert past_states[action_ix+1] != removed_shape
    #             x[action_ix, removed_shape] = 1
    #     return x.flatten()

    ACTION_IN_STATE_CONTEXT_DIM = NUM_TRANSITIONS
    def embed_action_in_state_context(self, action, state_before, state_after, past_states, past_actions):
        x = np.zeros(NUM_TRANSITIONS)

        if isinstance(action, Insert):
            inserted_shape = action.shape
            removed_at = set()

            for action_ix, (past_action, state) in enumerate(zip(past_actions, past_states)):
                if isinstance(past_action, Remove):
                    removed_ix = past_action.ix
                    removed_shape = state[removed_ix]
                    assert removed_shape in SHAPES
                    if action_ix < len(past_states) - 1:
                        assert past_states[action_ix+1] != removed_shape
                    if removed_shape == inserted_shape:
                        removed_at.add(action_ix)

            assert len(removed_at) != 0
            x[list(removed_at)] = 1
        return x
