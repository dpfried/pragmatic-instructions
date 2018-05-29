import util
import argparse
import scone.corpus
import json
import pandas

subjective_mappings = {
    'difficulty': {
        'very_hard': 5,
        'hard': 4,
        'not_so_hard': 3,
        'easy': 2,
        'very_easy': 1,
    },
    # 'undo': {
    #     'very_often': 5,
    #     'often': 4,
    #     'few_times': 3,
    #     'rarely': 2,
    #     'never': 1,
    # },
    'fluency': {
        'nonsense': 1,
        'major_errors': 2,
        'minor_errors': 3,
        'odd': 4,
        'acceptable': 5,
    },
    'amount_information': {
        'too_little': 1,
        'enough': 2,
        'too_much': 3,
    },
    'confidence': {
        'not_confident': 1,
        'slightly_unconfident': 2,
        'neutral': 3,
        'slightly_confident': 4,
        'confident': 5,
    },
}

def annotation_col(suffix):
    return 'Annotation.%s' % suffix


def load_turk_results(input_file):
    results_df = pandas.read_csv(input_file)
    return results_df[results_df['Answer.states-taken'] != '{}']


def add_states_correct(row_series, data_to_update, gold_instance):
    followed_states = json.loads(row_series['Answer.states-taken'])
    data_to_update[annotation_col('length_states_taken')] = len(followed_states)
    assert len(gold_instance.states) == scone.corpus.NUM_TRANSITIONS + 1
    if len(followed_states) != len(gold_instance.states):
        print(row_series)
        print(followed_states)
        print(gold_instance.states)
    assert len(followed_states) == len(gold_instance.states)
    # assert len(followed_states) >= len(gold_instance.states)
    for i, (fs, gs) in enumerate(zip(followed_states, gold_instance.states)):
        data_to_update[annotation_col('correct_at_%d' % i)] = (tuple(fs) == tuple(gs))


def add_subjective(row_series, data_to_update):
    for key in subjective_mappings.keys():
        data_to_update[annotation_col(key)] = subjective_mappings[key][row_series['Answer.%s' % key]]


def add_row_annotations(row_series, gold_instances_by_id):
    new_data = {}
    #
    gold_instance = gold_instances_by_id[row_series['Input.id']]
    add_states_correct(row_series, new_data, gold_instance)
    add_subjective(row_series, new_data)
    return row_series.append(pandas.Series(new_data))


def annotate(results_df, gold_instances):
    gold_instances_by_id = {
        instance.id: instance
        for instance in gold_instances
    }
    return results_df.apply(lambda row: add_row_annotations(row, gold_instances_by_id), axis=1)


def value_fractions(grouped):
    return grouped.value_counts() / grouped.count()

def run(args):
    corpus, train_instances, dev_instances, test_instances = scone.corpus.load_corpus_and_data(args)

    results = load_turk_results(args.input_results_file)

    results_annotated = annotate(results, test_instances)

    results_filtered = results_annotated[results_annotated[annotation_col('length_states_taken')] == scone.corpus.NUM_TRANSITIONS + 1]

    if args.ids_to_filter:
        results_filtered = results_filtered[results_filtered['Input.id'].isin(set(args.ids_to_filter))]

    source_grouped = results_filtered.groupby("Input.source")

    print(source_grouped.count())
    print()

    #print(value_fractions(source_grouped[annotation_col('correct_at_5')]))
    print("accuracy")
    print(source_grouped[annotation_col('correct_at_5')].mean())
    print()

    for key in subjective_mappings:
        print(key)
        named_values = subjective_mappings[key].items()
        print(min(named_values, key=lambda t: t[1]))
        print(max(named_values, key=lambda t: t[1]))
        print(source_grouped[annotation_col(key)].mean())
        print()

    print(value_fractions(source_grouped['Answer.who_generated']))
    print()

    return results_filtered, source_grouped

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_results_file")
    parser.add_argument("--ids_to_filter", nargs="*")

    scone.corpus.add_corpus_args(parser)

    return parser

if __name__ == "__main__":
    util.run(make_parser(), run)
