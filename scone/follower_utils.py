import dynet as dy

def random_gumbel(dim):
    return -dy.log(-dy.log(dy.random_uniform((dim,), 0.0, 1.0)))

def action_in_state_context_bonuses(corpus, state, inputs, W_context_action, W_action, predict_invalid, past_states, past_actions):
    all_inputs = dy.concatenate(inputs)
    bonuses = []
    # actions we're scoring could be all actions if we have an unconstrained model. so compute the valid actions for this corpus, and if we have an action that can't be applied in the state, just return a bonus of 0
    valid_actions = set(corpus.valid_actions(state))
    for action in corpus.ACTIONS:
        if action in valid_actions:
            next_state = corpus.take_action(state, action)
            embedded_action_sc = dy.inputVector(corpus.embed_action_in_state_context(action, state, next_state, past_states, past_actions))
            bonus = dy.dot_product(W_context_action * embedded_action_sc, all_inputs) + W_action * embedded_action_sc
        else:
            bonus = dy.scalarInput(0)
        bonuses.append(bonus)
    if predict_invalid:
        bonuses.append(dy.scalarInput(0))

    return dy.concatenate(bonuses)
