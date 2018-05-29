import dynet as dy
from scone.follower import ActionLayer

import scone.scene as scene

from .follower_utils import action_in_state_context_bonuses

class SceneFactored(ActionLayer):

    def __init__(self, corpus, model, init, input_dim, predict_invalid):
        super(SceneFactored, self).__init__(corpus, model, init, input_dim, predict_invalid)
        self._ACTION_TYPES = 6
        if predict_invalid:
            self._ACTION_TYPES += 1

        self.init_params(model, init)

    @property
    def ACTION_TYPES(self):
        try:
            return self._ACTION_TYPES
        except:
            self._ACTION_TYPES = 6
            return self._ACTION_TYPES

    def get_components(self):
        # return self.p_W_type, self.p_W_from, self.p_W_to, self.p_W_shirt, self.p_W_hat
        return self.p_W_type, self.p_W_from, self.p_W_to, self.p_W_shirt

    def restore_components(self, components):
        # self.p_W_type, self.p_W_from, self.p_W_to, self.p_W_shirt, self.p_W_hat = components
        self.p_W_type, self.p_W_from, self.p_W_to, self.p_W_shirt = components

    def init_params(self, model, init):
        self.p_W_type = model.parameters_from_numpy(init.initialize((self.ACTION_TYPES, self.input_dim)))
        self.p_W_from = model.parameters_from_numpy(init.initialize((scene.NUM_POSITIONS, self.input_dim)))
        self.p_W_to = model.parameters_from_numpy(init.initialize((scene.NUM_POSITIONS, self.input_dim)))
        self.p_W_shirt = model.parameters_from_numpy(init.initialize((scene.NUM_COLORS, self.input_dim)))
        # self.p_W_hat = model.parameters_from_numpy(init.initialize((scene.NUM_COLORS, self.input_dim)))

    def combine_logits(self, type_logits, from_logits, to_logits, shirt_logits):
        action_logits = []
        for action in self.corpus.ACTIONS:
            if isinstance(action, scene.Enter):
                assert 0 <= action.ix < scene.NUM_POSITIONS
                logt = type_logits[0] + from_logits[action.ix] + shirt_logits[scene.COLORS_TO_INDEX[action.shirt]]
            elif isinstance(action, scene.Move):
                assert 0 <= action.from_ix < scene.NUM_POSITIONS
                assert 0 <= action.to_ix < scene.NUM_POSITIONS
                logt = type_logits[1] + from_logits[action.from_ix] + to_logits[action.to_ix]
            elif isinstance(action, scene.Exit):
                assert 0 <= action.ix < scene.NUM_POSITIONS
                logt = type_logits[2] + from_logits[action.ix]
            elif isinstance(action, scene.SwitchPeople):
                assert 0 <= action.first_ix < scene.NUM_POSITIONS
                assert 0 <= action.second_ix < scene.NUM_POSITIONS
                logt = type_logits[3] + from_logits[action.first_ix] + to_logits[action.second_ix]
            elif isinstance(action, scene.TakeHat):
                assert 0 <= action.from_ix < scene.NUM_POSITIONS
                assert 0 <= action.to_ix < scene.NUM_POSITIONS
                logt = type_logits[4] + from_logits[action.from_ix] + to_logits[action.to_ix]
            elif isinstance(action, scene.NoOp):
                logt = type_logits[5]
            else:
                raise ValueError("invalid action type %s" % action)
            action_logits.append(logt)

        if self.predict_invalid:
            action_logits.append(type_logits[6])

        return dy.concatenate(action_logits)

    def compute_logits(self, input):
        W_type = dy.parameter(self.p_W_type)
        W_from = dy.parameter(self.p_W_from)
        W_to = dy.parameter(self.p_W_to)
        W_shirt = dy.parameter(self.p_W_shirt)
        # W_hat = dy.parameter(self.p_W_hat)

        type_logits = dy.log_softmax(W_type * input)
        from_logits = dy.log_softmax(W_from * input)
        to_logits = dy.log_softmax(W_to * input)
        shirt_logits = dy.log_softmax(W_shirt * input)
        # hat_logits = dy.log_softmax(W_hat * input)
        return type_logits, from_logits, to_logits, shirt_logits

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        assert len(inputs) == 1
        input = inputs[0]
        type_logits, from_logits, to_logits, shirt_logits = self.compute_logits(input)
        support = sorted([self.corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])

        unconstrained = self.combine_logits(type_logits, from_logits, to_logits, shirt_logits)

        return dy.log_softmax(unconstrained, support)

    @property
    def num_attention_heads(self):
        return 1

class SceneFactoredSplit(SceneFactored):
    def init_params(self, model, init):
        assert self.input_dim % 4 == 0
        self.p_dim = self.input_dim // 4
        self.p_W_type = model.parameters_from_numpy(init.initialize((self.ACTION_TYPES, self.p_dim)))
        self.p_W_beaker_from = model.parameters_from_numpy(init.initialize((scene.NUM_POSITIONS, self.p_dim)))
        self.p_W_beaker_to = model.parameters_from_numpy(init.initialize((scene.NUM_POSITIONS, self.p_dim)))
        self.p_W_shirt = model.parameters_from_numpy(init.initialize((scene.NUM_COLORS, self.p_dim)))
        # self.p_W_hat = model.parameters_from_numpy(init.initialize((scene.NUM_COLORS, p_dim)))

    def compute_logits(self, input):
        W_type = dy.parameter(self.p_W_type)
        W_from = dy.parameter(self.p_W_from)
        W_to = dy.parameter(self.p_W_to)
        W_shirt = dy.parameter(self.p_W_shirt)
        # W_hat = dy.parameter(self.p_W_hat)

        type_logits = dy.log_softmax(W_type * input[:self.p_dim])
        from_logits = dy.log_softmax(W_from * input[self.p_dim:2*self.p_dim])
        to_logits = dy.log_softmax(W_to * input[2*self.p_dim:3*self.p_dim])
        shirt_logits = dy.log_softmax(W_shirt * input[3*self.p_dim:])
        # hat_logits = dy.log_softmax(W_hat * input[3*self.p_dim:])

        return type_logits, from_logits, to_logits, shirt_logits

class SceneFactoredMultihead(SceneFactored):
    def _log_probs_unconstrained_unnormed(self, inputs, possible_actions):
        assert len(inputs) == self.num_attention_heads

        Ws = [dy.parameter(p) for p in [self.p_W_type, self.p_W_from, self.p_W_to, self.p_W_shirt]]

        type_logits, from_logits, to_logits, shirt_logits = [dy.log_softmax(W * inp)
                                                                         for (W, inp) in zip(Ws, inputs)]

        support = sorted([self.corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])
        unconstrained = self.combine_logits(type_logits, from_logits, to_logits, shirt_logits)
        return unconstrained, support

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        unconstrained, support = self._log_probs_unconstrained_unnormed(inputs, possible_actions)
        return dy.log_softmax(unconstrained, support)

    @property
    def num_attention_heads(self):
        return 4

    @property
    def attention_names(self):
        return ["type", "from_ix", "to_ix", "shirt"]

class SceneFactoredMultiheadContextual(SceneFactoredMultihead):
    def init_params(self, model, init):
        super(SceneFactoredMultiheadContextual, self).init_params(model, init)
        self.p_W_context_action = model.parameters_from_numpy(init.initialize((self.input_dim * self.num_attention_heads, self.corpus.ACTION_IN_STATE_CONTEXT_DIM)))
        self.p_W_action = model.parameters_from_numpy(init.initialize((1, self.corpus.ACTION_IN_STATE_CONTEXT_DIM)))

    def get_components(self):
        comps = super(SceneFactoredMultiheadContextual, self).get_components()
        comps += (self.p_W_context_action, self.p_W_action)
        return comps

    def restore_components(self, components):
        k = 2
        super(SceneFactoredMultiheadContextual, self).restore_components(components[:-k])
        self.p_W_context_action, self.p_W_action = components[-k:]

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        assert state is not None
        W_context_action = dy.parameter(self.p_W_context_action)
        W_action = dy.parameter(self.p_W_action)
        unconstrained, support = self._log_probs_unconstrained_unnormed(inputs, possible_actions)
        unconstrained += action_in_state_context_bonuses(self.corpus, state, inputs, W_context_action, W_action, self.predict_invalid, past_states, past_actions)
        return dy.log_softmax(unconstrained, support)
