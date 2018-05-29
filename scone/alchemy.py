from .corpus import Corpus, INVALID
from collections import namedtuple
import numpy as np

Mix = namedtuple("Mix", ["ix", "t"])

def mix(ix):
    return Mix(ix=ix, t="alc_m")

Drain = namedtuple("Drain", ["amt", "ix", "t"])

def drain(amt, ix):
    return Drain(amt=amt, ix=ix, t="alc_d")

Pour = namedtuple("Pour", ["from_ix", "to_ix", "t"])

def pour(from_ix, to_ix):
    return Pour(from_ix=from_ix, to_ix=to_ix, t="alc_p")

NUM_BEAKERS = 7

BEAKER_SIZE = 4

BEAKER_INDICES = range(NUM_BEAKERS)

COLORS = ['b', 'g', 'o', 'p', 'r', 'y']
COLORS_TO_INDEX = {
    color: index
    for (index, color) in enumerate(sorted(COLORS))
}
NUM_COLORS = len(COLORS)


class AlchemyCorpus(Corpus):

    ACTIONS = \
        [
            drain(amount, beaker_ix)
            for amount in range(1, BEAKER_SIZE + 1)
            for beaker_ix in BEAKER_INDICES
        ] + [
            mix(beaker_ix)
            for beaker_ix in BEAKER_INDICES
        ] + [
            pour(from_beaker_ix, to_beaker_ix)
            for from_beaker_ix in BEAKER_INDICES
            for to_beaker_ix in BEAKER_INDICES
            if from_beaker_ix != to_beaker_ix
        ]

    def dataset_name(self):
        return "alchemy"

    def parse_token(self, token):
        ix, content = token.split(":")
        return content if content != "_" else ""


    def parse_state(self, state_string):
        toks = state_string.split()
        assert len(toks) == NUM_BEAKERS
        return tuple(self.parse_token(token) for token in toks)


    def valid_actions(self, state):
        def it():
            for beaker_ix in BEAKER_INDICES:
                contents = state[beaker_ix]
                if len(contents) == 0:
                    continue

                if not(all(c == contents[0] for c in contents[1:])):
                    yield mix(beaker_ix)

                for amt in range(1, len(contents) + 1):
                    yield drain(amt, beaker_ix)

                for other_ix in BEAKER_INDICES:
                    if other_ix == beaker_ix:
                        continue
                    if len(state[other_ix]) + len(contents) <= BEAKER_SIZE:
                        yield pour(beaker_ix, other_ix)
        return list(it())


    def take_action(self, state, action):
        new_state = list(state)

        if isinstance(action, Mix):
            assert new_state[action.ix]
            new_state[action.ix] = 'b' * len(new_state[action.ix])
        elif isinstance(action, Drain):
            assert len(new_state[action.ix]) >= action.amt
            new_state[action.ix] = new_state[action.ix][:-action.amt]
        elif isinstance(action, Pour):
            assert len(new_state[action.from_ix]) + len(new_state[action.to_ix]) <= BEAKER_SIZE
            new_state[action.to_ix] += new_state[action.from_ix][::-1] # reverse when pouring
            new_state[action.from_ix] = ''
        elif action == INVALID:
            return None
        else:
            raise ValueError("bad action type", action)

        return tuple(new_state)


    STATE_DIM = NUM_BEAKERS * BEAKER_SIZE * NUM_COLORS

    def _embed_state_notflat(self, state):
        assert len(state) == NUM_BEAKERS

        x = np.zeros((NUM_BEAKERS, BEAKER_SIZE, NUM_COLORS))

        for beaker_ix, beaker_contents in enumerate(state):
            assert len(beaker_contents) <= BEAKER_SIZE
            for content_ix, color in enumerate(beaker_contents):
                x[beaker_ix,content_ix,COLORS_TO_INDEX[color]] = 1
        return x

    def embed_state(self, state):
        return self._embed_state_notflat(state).flatten()

    ACTION_DIM = 3 + NUM_BEAKERS + NUM_BEAKERS + BEAKER_SIZE

    def embed_action(self, action):
        # [ActionType FromIx ToIx Amount]
        action_type = np.zeros(3)
        from_ix = np.zeros(NUM_BEAKERS)
        to_ix = np.zeros(NUM_BEAKERS)
        amt = np.zeros(BEAKER_SIZE)
        if isinstance(action, Mix):
            action_type[0] = 1
            to_ix[action.ix] = 1
        elif isinstance(action, Drain):
            action_type[1] = 1
            to_ix[action.ix] = 1
            amt[action.amt - 1] = 1
        elif isinstance(action, Pour):
            action_type[2] = 1
            from_ix[action.from_ix] = 1
            to_ix[action.to_ix] = 1
        else:
            raise ValueError("bad action type", action)
        return np.hstack([action_type, from_ix, to_ix, amt])

    ACTION_IN_STATE_CONTEXT_DIM = 2 * BEAKER_SIZE * NUM_COLORS
    def embed_action_in_state_context(self, action, state_before, state_after, past_states, past_actions):
        emb_state = self._embed_state_notflat(state_before)
        beaker_contents = np.zeros((2, BEAKER_SIZE, NUM_COLORS))
        assert beaker_contents.shape[1:] == emb_state.shape[1:]
        if isinstance(action, Mix) or isinstance(action, Drain):
            beaker_contents[1] = emb_state[action.ix]
        elif isinstance(action, Pour):
            beaker_contents[0] = emb_state[action.from_ix]
            beaker_contents[1] = emb_state[action.to_ix]
        else:
            raise ValueError("bad action type", action)
        return beaker_contents.flatten()
