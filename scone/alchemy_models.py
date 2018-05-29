import dynet as dy
from scone.follower import ActionLayer

import scone.alchemy as alchemy

from .follower_utils import action_in_state_context_bonuses

class AlchemyFactored(ActionLayer):

    def __init__(self, corpus, model, init, input_dim, predict_invalid):
        super(AlchemyFactored, self).__init__(corpus, model, init, input_dim, predict_invalid)
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
        return self.p_W_type, self.p_W_beaker_from, self.p_W_beaker_to, self.p_W_amount

    def restore_components(self, components):
        self.p_W_type, self.p_W_beaker_from, self.p_W_beaker_to, self.p_W_amount = components

    def init_params(self, model, init):
        self.p_W_type = model.parameters_from_numpy(init.initialize((self.ACTION_TYPES, self.input_dim)))
        self.p_W_beaker_from = model.parameters_from_numpy(init.initialize((alchemy.NUM_BEAKERS, self.input_dim)))
        self.p_W_beaker_to = model.parameters_from_numpy(init.initialize((alchemy.NUM_BEAKERS, self.input_dim)))
        self.p_W_amount = model.parameters_from_numpy(init.initialize((alchemy.BEAKER_SIZE, self.input_dim)))

    def combine_logits(self, type_logits, beaker_from_logits, beaker_to_logits, amount_logits):
        action_logits = []
        for action in self.corpus.ACTIONS:
            if isinstance(action, alchemy.Drain):
                assert 1 <= action.amt <= alchemy.BEAKER_SIZE
                assert 0 <= action.ix < alchemy.NUM_BEAKERS
                logt = type_logits[0] + beaker_from_logits[action.ix] + amount_logits[action.amt - 1]
            elif isinstance(action, alchemy.Mix):
                assert 0 <= action.ix < alchemy.NUM_BEAKERS
                logt = type_logits[1] + beaker_from_logits[action.ix]
            elif isinstance(action, alchemy.Pour):
                assert 0 <= action.from_ix < alchemy.NUM_BEAKERS
                assert 0 <= action.to_ix < alchemy.NUM_BEAKERS
                logt = type_logits[2] + beaker_from_logits[action.from_ix] + beaker_to_logits[action.to_ix]
            else:
                raise ValueError("invalid action type %s" % action)
            action_logits.append(logt)

        if self.predict_invalid:
            action_logits.append(type_logits[3])

        return dy.concatenate(action_logits)

    def compute_logits(self, input):
        W_type = dy.parameter(self.p_W_type)
        W_beaker_from = dy.parameter(self.p_W_beaker_from)
        W_beaker_to = dy.parameter(self.p_W_beaker_to)
        W_amount = dy.parameter(self.p_W_amount)

        type_logits = dy.log_softmax(W_type * input)
        beaker_from_logits = dy.log_softmax(W_beaker_from * input)
        beaker_to_logits = dy.log_softmax(W_beaker_to * input)
        amount_logits = dy.log_softmax(W_amount * input)
        return type_logits, beaker_from_logits, beaker_to_logits, amount_logits

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        assert len(inputs) == 1
        input = inputs[0]
        type_logits, beaker_from_logits, beaker_to_logits, amount_logits = self.compute_logits(input)
        support = sorted([self.corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])

        unconstrained = self.combine_logits(type_logits, beaker_from_logits, beaker_to_logits, amount_logits)

        return dy.log_softmax(unconstrained, support)

    @property
    def num_attention_heads(self):
        return 1

class AlchemyFactoredSplit(AlchemyFactored):
    def init_params(self, model, init):
        assert self.input_dim % 4 == 0
        p_dim = self.input_dim // 4
        self.p_W_type = model.parameters_from_numpy(init.initialize((self.ACTION_TYPES, p_dim)))
        self.p_W_beaker_from = model.parameters_from_numpy(init.initialize((alchemy.NUM_BEAKERS, p_dim)))
        self.p_W_beaker_to = model.parameters_from_numpy(init.initialize((alchemy.NUM_BEAKERS, p_dim)))
        self.p_W_amount = model.parameters_from_numpy(init.initialize((alchemy.BEAKER_SIZE, p_dim)))

    def compute_logits(self, input):
        W_type = dy.parameter(self.p_W_type)
        W_beaker_from = dy.parameter(self.p_W_beaker_from)
        W_beaker_to = dy.parameter(self.p_W_beaker_to)
        W_amount = dy.parameter(self.p_W_amount)

        p_dim = self.input_dim // 4

        type_logits = dy.log_softmax(W_type * input[:p_dim])
        beaker_from_logits = dy.log_softmax(W_beaker_from * input[p_dim:2*p_dim])
        beaker_to_logits = dy.log_softmax(W_beaker_to * input[2*p_dim:3*p_dim])
        amount_logits = dy.log_softmax(W_amount * input[3*p_dim:])

        return type_logits, beaker_from_logits, beaker_to_logits, amount_logits


class AlchemyFactoredMultihead(AlchemyFactored):
    def _log_probs_unconstrained_unnormed(self, inputs, possible_actions):
        assert len(inputs) == self.num_attention_heads

        Ws = [dy.parameter(p) for p in [self.p_W_type, self.p_W_beaker_from, self.p_W_beaker_to, self.p_W_amount]]

        type_logits, beaker_from_logits, beaker_to_logits, amount_logits = [dy.log_softmax(W * inp)
                                                                       for (W, inp) in zip(Ws, inputs)]

        support = sorted([self.corpus.ACTIONS_TO_INDEX[ac] for ac in possible_actions])
        unconstrained = self.combine_logits(type_logits, beaker_from_logits, beaker_to_logits, amount_logits)
        return unconstrained, support

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        unconstrained, support = self._log_probs_unconstrained_unnormed(inputs, possible_actions)
        return dy.log_softmax(unconstrained, support)

    @property
    def num_attention_heads(self):
        return 4

    @property
    def attention_names(self):
        return ["type", "from_ix", "to_ix", "amt"]


class AlchemyFactoredMultiheadContextual(AlchemyFactoredMultihead):
    def init_params(self, model, init):
        super(AlchemyFactoredMultiheadContextual, self).init_params(model, init)
        self.p_W_context_action = model.parameters_from_numpy(init.initialize((self.input_dim * self.num_attention_heads, self.corpus.ACTION_IN_STATE_CONTEXT_DIM)))
        self.p_W_action = model.parameters_from_numpy(init.initialize((1, self.corpus.ACTION_IN_STATE_CONTEXT_DIM)))

    def get_components(self):
        comps = super(AlchemyFactoredMultiheadContextual, self).get_components()
        comps += (self.p_W_context_action, self.p_W_action)
        return comps

    def restore_components(self, components):
        k = 2
        super(AlchemyFactoredMultiheadContextual, self).restore_components(components[:-k])
        self.p_W_context_action, self.p_W_action = components[-k:]

    def compute_output_log_probs(self, inputs, possible_actions, state=None, past_states=None, past_actions=None):
        assert state is not None
        W_context_action = dy.parameter(self.p_W_context_action)
        W_action = dy.parameter(self.p_W_action)
        unconstrained, support = self._log_probs_unconstrained_unnormed(inputs, possible_actions)
        unconstrained += action_in_state_context_bonuses(self.corpus, state, inputs, W_context_action, W_action, self.predict_invalid, past_states=past_states, past_actions=past_actions)
        return dy.log_softmax(unconstrained, support)
