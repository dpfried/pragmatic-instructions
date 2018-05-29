import re
import json
import argparse

def get_stats(fname):
    print(fname)
    max_acc = None
    argmax_acc = None

    params = {}

    read_params = False
    with open(fname) as f:
        for line in f:
            if not read_params and line.startswith("{"):
                params = json.loads(line)
                read_params = True
            if not read_params:
                continue
            match = re.match(r"(.*) dev_beam end state.*\((.*)\)%", line)
            if not match:
                match = re.match(r"(.*)\s+dev_beam bleu:\s*([\d\.]+)", line)
            if match:
                epoch, acc = match.groups()
                epoch = int(epoch)
                acc = float(acc)
                if max_acc is None or acc > max_acc:
                    max_acc = acc
                    argmax_acc = epoch

    params['best_dev_eval'] = max_acc
    params['best_dev_epoch'] = argmax_acc
    return params

if __name__ == "__main__":
    import sys
    import glob
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    all_params = []
    for arg in args.files:
        for fname in glob.glob(arg):
            params = get_stats(fname)
            params['filename'] = fname
            print("%s\t%s\t%s" % (fname, params['best_dev_eval'], params['best_dev_epoch']))
            all_params.append(params)

    if args.output_file:
        import pandas
        pandas.DataFrame(all_params).to_csv(args.output_file)
