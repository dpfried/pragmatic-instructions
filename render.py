import json
import scone.follower
import json
import textwrap
import base64

def analyze(instance_and_predicted):
    inst, (lit_states, lit_actions), (prag_states, prag_actions) = instance_and_predicted
    print("literal")
    scone.follower.compare_prediction(None, inst, lit_states, lit_actions, None)
    print("\npragmatic")
    scone.follower.compare_prediction(None, inst, prag_states, prag_actions, None)

def render_pragmatic_follower(corpus_name, instance_and_predicted):
    inst, (lit_states, lit_actions), (prag_states, prag_actions) = instance_and_predicted
    data = []
    for name, states in [("gold", inst.states), ("literal", lit_states), ("pragmatic", prag_states)]:
        data.append({
            'corpus': corpus_name,
            'name': inst.id + ": " + name,
            'sequence': collapse(states, inst.utterances)
        })
    return base64.b64encode(json.dumps(data).encode('utf-8')).decode('utf-8')

def render_pragmatic_speaker(corpus_name, instance_and_predicted):
    inst, lit_utterances, prag_utterances = instance_and_predicted
    data = []
    for name, utts in [("gold", inst.utterances), ("literal", lit_utterances), ("pragmatic", prag_utterances)]:
        data.append({
            'corpus': corpus_name,
            'name': inst.id + ": " + name,
            'sequence': collapse(inst.states, utts)
        })
    return base64.b64encode(json.dumps(data).encode('utf-8')).decode('utf-8')

def collapse(states, utterances):
    def render_state(state):
        return ' '.join("%d:%s" % (i+1, s) for (i, s) in enumerate(state))
    collapsed = []
    for i in range(len(utterances)):
        collapsed.append(render_state(states[i]))
        collapsed.append(textwrap.fill(' '.join(utterances[i]), 40))
    collapsed.append(render_state(states[-1]))
    return collapsed
