import util
import argparse
import scone.corpus
import json
import pandas

from collections import namedtuple

DumpableInstance = namedtuple("DumpableInstance", ["id", "source", "utterances", "initial_state"])

def dump_to_json(dumpable_instances, output_file, instance_ids_to_take):
    df = pandas.DataFrame([
        {
            'id': di.id,
            'source': di.source,
            'utterances': json.dumps([' '.join(utt).replace("'", "&apos;") for utt in di.utterances]),
            'initial_state': json.dumps(di.initial_state)
        }
        for di in dumpable_instances
        if di.id in instance_ids_to_take
    ])
    #df.sample(frac=1).to_csv(output_file, index=False)
    df.sort_values('source').to_csv(output_file, index=False)

def load_dumpable_instances_from_speaker_file(input_file, gold_instances, system_name, append_gold=True):
    with open(input_file) as f:
        system_instances = json.load(f)

    gold_instances_by_id = {
        instance.id: instance
        for instance in gold_instances
    }

    dumpable_instances = []

    system_ids = set()

    for system_data in system_instances:
        follower_weight = system_data['follower_weight']
        source = "%s-fw=%s" % (system_name, follower_weight)
        for id, utterances in system_data['predictions'].items():
            system_ids.add(id)
            dumpable_instances.append(DumpableInstance(
                id=id, source=source, utterances=utterances, initial_state=gold_instances_by_id[id].states[0]
            ))

    if append_gold:
        for instance in gold_instances:
            if instance.id not in system_ids:
                continue
            dumpable_instances.append(DumpableInstance(
                id=instance.id, source="human", utterances=instance.utterances, initial_state=instance.states[0]
            ))

    return dumpable_instances

def run(args):
    assert len(args.input_files_from_rational_speaker) == len(args.system_names)
    corpus, train_instances, dev_instances, test_instances = scone.corpus.load_corpus_and_data(args)

    instance_ids = list(sorted(set([inst.id for inst in test_instances])))
    import random
    random.seed(1)
    random.shuffle(instance_ids)

    instance_ids_to_take = instance_ids[:args.num_instances]

    dumpable_instances = [
        di
        for i, (fname, system_name) in enumerate(zip(args.input_files_from_rational_speaker, args.system_names))
        for di in load_dumpable_instances_from_speaker_file(fname, dev_instances + test_instances, system_name, append_gold=i==0)
    ]

    dump_to_json(dumpable_instances, args.output_file, instance_ids_to_take)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files_from_rational_speaker", nargs="+", help="*.predictions.json, from rational_speaker")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--system_names", nargs="+", default=["speaker"], help="must be one for each file in input_files_from_rational_speaker")

    scone.corpus.add_corpus_args(parser)

    return parser

if __name__ == "__main__":
    util.run(make_parser(), run)
