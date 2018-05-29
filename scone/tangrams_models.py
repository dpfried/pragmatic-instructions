import dynet as dy
from scone.follower import ActionLayer

import scone.tangrams as tangrams

from .follower_utils import action_in_state_context_bonuses

class TangramsFactored(ActionLayer):

    def __init__(self, corpus, model, init, input_dim, predict_invalid):
        super(TangramsFactored, self).__init__(corpus, model, init, input_dim, predict_invalid)
        self._ACTION_TYPES = 3
        if self.predict_invalid:
            self._ACTION_TYPES += 1
        self.init_params(model, init)

    @property
    def ACTION_TYPES(self):
        try:
            return self._ACTION_TYPES
        except:
            self._ACTION_TYPES = 3
            return self._ACTION_TYPES

    def get_components(self):
        return self.p_W_type, self.p_W_first_ix, self.p_W_second_ix, self.p_W_shape

    def restore_components(self, components):
        self.p_W_type, self.p_W_first_ix, self.p_W_second_ix, self.p_W_shape = components

    def init_params(self, model, init):
        self.p_W_type = model.parameters_from_numpy(init.initialize((self.ACTION_TYPES, self.input_dim)))
        self.p_W_first_ix = model.parameters_from_numpy(init.initialize((tangrams.MAX_POSITIONS, self.input_dim)))
        self.p_W_second_ix = model.parameters_from_numpy(init.initialize((tangrams.MAX_POSITIONS, self.input_dim)))
        self.p_W_shape = model.parameters_from_numpy(init.initialize((tangrams.NUM_SHAPES, self.input_dim)))

    def combine_logits(self, type_logits, first_ix_logits, second_ix_logits, shape_logits):
        action_logits = []
        for action in self.corpus.ACTIONS:
            if isinstance(action, tangrams.Insert):
                assert 0 <= action.shape < tangrams.NUM_SHAPES
                assert 0 <= action.ix < tangrams.MAX_POSITIONS
                logt = type_logits[0] + first_ix_logits[action.ix] + shape_logits[action.shape]
            elif isinstance(action, tangrams.Remove):
                assert 0 <= action.ix < tangrams.MAX_POSITIONS
                logt = type_logits[1] + first_ix_logits[action.ix]
            elif isinstance(action, tangrams.Swap):
                assert 0 <= action.first_ix < tangrams.MAX_POSITIONS
                assert 0 <= action.second_ix < tangrams.MAX_POSITIONS
                logt = type_logits[2] + first_ix_logits[action.first_ix] + second_ix_logits[action.second_ix]
            else:
                raise ValueError("invalid action type %s" % action)
            action_logits.append(logt)

        if self.predict_invalid:
            action_logits.append(type_logits[3])

        return dy.concatenate(action_logits)

    def compute_logits(self, input):
        W_type = dy.parameter(self.p_W_type)
        W_first_ix = dy.parameter(self.p_W_first_ix)
        W_second_ix = dy.parameter(self.p_W_second_ix)
        W_shape = dy.parameter(self.p_W_shape)

        type_logits = dy.log_softmax(W_type * input)
        first_ix_logits = dy.log_softmax(W_first_ix * input)
        second_ix_logits = dy.log_softmax(W_second_ix * input)
        shape_logits = dy.log_softmax(W_shape * input)
        return type_logits, first_ix_logits, second_ix_logits, shape_logits

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        assert len(inputs) == 1
        input = inputs[0]
        type_logits, first_ix_logits, second_ix_logits, shape_logits = self.compute_logits(input)
        support = sorted([self.corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])
        if self.predict_invalid:
            support += self.corpus.INVALID_INDEX

        unconstrained = self.combine_logits(type_logits, first_ix_logits, second_ix_logits, shape_logits)

        return dy.log_softmax(unconstrained, support)

    @property
    def num_attention_heads(self):
        return 1

class TangramsFactoredSplit(TangramsFactored):
    def init_params(self, model, init):
        assert self.input_dim % 4 == 0
        self.p_dim = self.input_dim // 4
        self.p_W_type = model.parameters_from_numpy(init.initialize((self.ACTION_TYPES, self.p_dim)))
        self.p_W_first_ix = model.parameters_from_numpy(init.initialize((tangrams.MAX_POSITIONS, self.p_dim)))
        self.p_W_second_ix = model.parameters_from_numpy(init.initialize((tangrams.MAX_POSITIONS, self.p_dim)))
        self.p_W_shape = model.parameters_from_numpy(init.initialize((tangrams.NUM_SHAPES, self.p_dim)))

    def compute_logits(self, input):
        W_type = dy.parameter(self.p_W_type)
        W_first_ix = dy.parameter(self.p_W_first_ix)
        W_second_ix = dy.parameter(self.p_W_second_ix)
        W_shape = dy.parameter(self.p_W_shape)

        type_logits = dy.log_softmax(W_type * input[:self.p_dim])
        first_ix_logits = dy.log_softmax(W_first_ix * input[self.p_dim:2*self.p_dim])
        second_ix_logits = dy.log_softmax(W_second_ix * input[2*self.p_dim:3*self.p_dim])
        shape_logits = dy.log_softmax(W_shape * input[3*self.p_dim:])
        return type_logits, first_ix_logits, second_ix_logits, shape_logits


class TangramsFactoredMultihead(TangramsFactored):
    def _log_probs_unconstrained_unnormed(self, inputs, possible_actions):
        assert len(inputs) == self.num_attention_heads

        Ws = [dy.parameter(p) for p in [self.p_W_type, self.p_W_first_ix, self.p_W_second_ix, self.p_W_shape]]

        type_logits, first_ix_logits, second_ix_logits, shape_logits = [dy.log_softmax(W * inp)
                                                                            for (W, inp) in zip(Ws, inputs)]

        support = sorted([self.corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])
        unconstrained = self.combine_logits(type_logits, first_ix_logits, second_ix_logits, shape_logits)
        return unconstrained, support

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        unconstrained, support = self._log_probs_unconstrained_unnormed(inputs, possible_actions)
        return dy.log_softmax(unconstrained, support)

    @property
    def num_attention_heads(self):
        return 4

    @property
    def attention_names(self):
        return ["type", "fst_i", "snd_i", "shape"]


class TangramsFactoredMultiheadContextual(TangramsFactoredMultihead):
    def init_params(self, model, init):
        super(TangramsFactoredMultiheadContextual, self).init_params(model, init)
        self.p_W_context_action = model.parameters_from_numpy(init.initialize((self.input_dim * self.num_attention_heads, self.corpus.ACTION_IN_STATE_CONTEXT_DIM)))
        self.p_W_action = model.parameters_from_numpy(init.initialize((1, self.corpus.ACTION_IN_STATE_CONTEXT_DIM)))

    def get_components(self):
        comps = super(TangramsFactoredMultiheadContextual, self).get_components()
        comps += (self.p_W_context_action, self.p_W_action)
        return comps

    def restore_components(self, components):
        k = 2
        super(TangramsFactoredMultiheadContextual, self).restore_components(components[:-k])
        self.p_W_context_action, self.p_W_action = components[-k:]

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        assert state is not None
        W_context_action = dy.parameter(self.p_W_context_action)
        W_action = dy.parameter(self.p_W_action)
        unconstrained, support = self._log_probs_unconstrained_unnormed(inputs, possible_actions)
        unconstrained += action_in_state_context_bonuses(self.corpus, state, inputs, W_context_action, W_action, self.predict_invalid, past_states, past_actions)
        return dy.log_softmax(unconstrained, support)
