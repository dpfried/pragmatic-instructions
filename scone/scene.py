from .corpus import Corpus, INVALID
from collections import namedtuple

import numpy as np

Enter = namedtuple("Enter", ["ix", "shirt", "t"])

def enter(ix, shirt):
    return Enter(ix=ix, shirt=shirt, t="scene_en")

Move = namedtuple("Move", ["from_ix", "to_ix", "t"])

def move(from_ix, to_ix):
    return Move(from_ix=from_ix, to_ix=to_ix, t="scene_m")

Exit = namedtuple("Exit", ["ix", "t"])

def exit(ix):
    return Exit(ix=ix, t="scene_ex")

SwitchPeople = namedtuple("SwitchPeople", ["first_ix", "second_ix", "t"])

def switch_people(first_ix, second_ix):
    return SwitchPeople(first_ix=first_ix, second_ix=second_ix, t="scene_ppl_s")

TakeHat = namedtuple("TakeHat", ["from_ix", "to_ix", "t"])

def take_hat(from_ix, to_ix):
    return TakeHat(from_ix=from_ix, to_ix=to_ix, t="scene_take_hat")

NoOp = namedtuple("NoOp", ["t"])

def no_op():
    return NoOp("scene_noop")

NUM_POSITIONS = 10

POSITION_INDICES = range(NUM_POSITIONS)

COLORS = ['_', 'b', 'g', 'o', 'p', 'r', 'y']
COLORS_TO_INDEX = {
    color: index
    for (index, color) in enumerate(sorted(COLORS))
}
BLANK_COLOR = '_'
BLANK_COLOR_INDEX = COLORS.index(BLANK_COLOR)

DOUBLE_BLANK = BLANK_COLOR + BLANK_COLOR

NUM_COLORS = len(COLORS)

class SceneCorpus(Corpus):
    ACTIONS = \
        [
            enter(ix, shirt)
            for ix in POSITION_INDICES
            for shirt in COLORS
            if shirt != BLANK_COLOR
        ] + [
            move(from_ix, to_ix)
            for from_ix in POSITION_INDICES
            for to_ix in POSITION_INDICES
            if from_ix != to_ix
        ] + [
            exit(ix)
            for ix in POSITION_INDICES
        ] + [
            switch_people(from_ix, to_ix)
            for from_ix in POSITION_INDICES
            for to_ix in POSITION_INDICES
            if from_ix - to_ix == 1 or to_ix - from_ix == 1
        ] + [
            take_hat(from_ix, to_ix)
            for from_ix in POSITION_INDICES
            for to_ix in POSITION_INDICES
            if from_ix - to_ix == 1 or to_ix - from_ix == 1
        ] + [
            no_op()
        ]


    def dataset_name(self):
        return "scene"


    def parse_token(self, token):
        ix, content = token.split(":")
        assert len(content) == 2 and content[0] in COLORS and content[1] in COLORS
        return content


    def parse_state(self, state_string):
        toks = state_string.split()
        assert len(toks) == NUM_POSITIONS
        return tuple(self.parse_token(token) for token in toks)


    def valid_actions(self, state):
        def at_anchor_point(ix):
            return ix == 0 or ix == NUM_POSITIONS - 1 or (ix > 0 and state[ix-1][0] != BLANK_COLOR) or (ix < NUM_POSITIONS - 1 and state[ix+1][0] != BLANK_COLOR)
        def it():
            for ix in POSITION_INDICES:
                if state[ix][0] == BLANK_COLOR:
                    # ENTER
                    if at_anchor_point(ix):
                        for shirt_color in COLORS:
                            if shirt_color == BLANK_COLOR:
                                continue
                            yield enter(ix, shirt_color)
                else:
                    # EXIT
                    yield exit(ix)
                    for to_ix in POSITION_INDICES:
                        if to_ix == ix:
                            continue
                        if at_anchor_point(to_ix) and state[to_ix][0] == BLANK_COLOR:
                            # MOVE
                            yield move(ix, to_ix)
                        else:
                            # SwitchPeople
                            if ix - to_ix == 1 or to_ix - ix == 1:
                                yield switch_people(ix, to_ix)
                                if state[ix][1] != BLANK_COLOR and state[to_ix][1] == BLANK_COLOR:
                                    yield take_hat(ix, to_ix)
            yield no_op()
        return list(it())


    def take_action(self, state, action):
        new_state = list(state)
        if isinstance(action, Enter):
            assert new_state[action.ix][0] == BLANK_COLOR
            new_state[action.ix] = action.shirt + BLANK_COLOR
        elif isinstance(action, Move):
            assert new_state[action.from_ix][0] != BLANK_COLOR and new_state[action.to_ix][0] == BLANK_COLOR
            new_state[action.to_ix] = new_state[action.from_ix]
            new_state[action.from_ix] = DOUBLE_BLANK
        elif isinstance(action, Exit):
            assert new_state[action.ix][0] != BLANK_COLOR
            new_state[action.ix] = DOUBLE_BLANK
        elif isinstance(action, SwitchPeople):
            assert new_state[action.first_ix][0] != BLANK_COLOR and new_state[action.second_ix][0] != BLANK_COLOR
            new_state[action.first_ix], new_state[action.second_ix] = new_state[action.second_ix], new_state[action.first_ix]
        elif isinstance(action, TakeHat):
            assert new_state[action.from_ix][1] != BLANK_COLOR and new_state[action.to_ix][1] == BLANK_COLOR
            new_from = new_state[action.from_ix][0] + BLANK_COLOR
            new_to = new_state[action.to_ix][0] + new_state[action.from_ix][1]
            new_state[action.from_ix] = new_from
            new_state[action.to_ix] = new_to
        elif isinstance(action, NoOp):
            pass
        elif action == INVALID:
            return None
        else:
            raise ValueError("bad action type", action)
        return tuple(new_state)

    STATE_DIM = NUM_POSITIONS * 2 * (NUM_COLORS)

    def _embed_state_notflat(self, state):
        assert len(state) == NUM_POSITIONS

        x = np.zeros((NUM_POSITIONS, 2, NUM_COLORS))

        for ix, contents in enumerate(state):
            assert len(contents) == 2
            shirt, hat = contents[0], contents[1]
            x[ix, 0, COLORS_TO_INDEX[shirt]] = 1
            x[ix, 1, COLORS_TO_INDEX[hat]] = 1
        return x

    def embed_state(self, state):
        return self._embed_state_notflat(state).flatten()

    ACTION_DIM = 6 + NUM_COLORS + 2 * NUM_POSITIONS

    def embed_action(self, action):
        action_type = np.zeros(6)
        from_ix = np.zeros(NUM_POSITIONS)
        to_ix = np.zeros(NUM_POSITIONS)
        shirt = np.zeros(NUM_COLORS)
        if isinstance(action, Enter):
            action_type[0] = 1
            from_ix[action.ix] = 1
            shirt[COLORS_TO_INDEX[action.shirt]] = 1
        elif isinstance(action, Move):
            action_type[1] = 1
            from_ix[action.from_ix] = 1
            to_ix[action.to_ix] = 1
        elif isinstance(action, Exit):
            action_type[2] = 1
            from_ix[action.ix] = 1
        elif isinstance(action, SwitchPeople):
            action_type[3] = 1
            from_ix[action.first_ix] = 1
            to_ix[action.second_ix] = 1
        elif isinstance(action, TakeHat):
            action_type[4] = 1
            from_ix[action.from_ix] = 1
            to_ix[action.to_ix] = 1
        elif isinstance(action, NoOp):
            action_type[5] = 1
        else:
            raise ValueError()
        return np.hstack([action_type, from_ix, to_ix, shirt])

    ACTION_IN_STATE_CONTEXT_DIM = 4 * (2 * NUM_COLORS)
    def embed_action_in_state_context(self, action, state_before, state_after, past_states, past_actions):
        emb_state_before = self._embed_state_notflat(state_before)
        emb_state_after = self._embed_state_notflat(state_before)
        persons_affected = np.zeros((2, 2, NUM_COLORS))
        anchor_points = np.zeros((2, 2, NUM_COLORS))
        assert persons_affected.shape[1:] == emb_state_before.shape[1:]
        assert anchor_points.shape[1:] == emb_state_before.shape[1:]
        if isinstance(action, Enter):
            persons_affected[0] = emb_state_after[action.ix]
            anchor_ix = action.ix
        elif isinstance(action, Exit):
            persons_affected[0] = emb_state_before[action.ix]
            anchor_ix = action.ix
        elif isinstance(action, Move):
            persons_affected[0] = emb_state_before[action.from_ix]
            anchor_ix = action.to_ix
        elif isinstance(action, TakeHat):
            persons_affected[0] = emb_state_before[action.from_ix]
            persons_affected[1] = emb_state_after[action.to_ix]
            anchor_ix = None
        elif isinstance(action, SwitchPeople):
            persons_affected[0] = emb_state_before[action.first_ix]
            persons_affected[1] = emb_state_after[action.second_ix]
            anchor_ix = None
        elif isinstance(action, NoOp):
            anchor_ix = None
        else:
            raise ValueError("bad action type", action)
        if anchor_ix is not None:
            if anchor_ix > 0:
                # use state after so that we are not anchoring to the person themself if they moved one left or right
                anchor_points[0] = emb_state_after[anchor_ix - 1]
            if anchor_ix < NUM_POSITIONS - 1:
                anchor_points[1] = emb_state_after[anchor_ix + 1]
        return np.hstack([persons_affected.flatten(), anchor_points.flatten()])
