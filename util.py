__author__ = 'dfried'
import numpy as np
import pickle
from collections import namedtuple

import sys
import subprocess

import keyword

import os
import json

class GlorotInitializer(object):
    def __init__(self, seed):
        self.seed = seed
        self.state = np.random.RandomState(seed)

    def initialize(self, dimensions, lookup=False):
        # TODO: cropping the right dimension?
        dims = dimensions[1:] if lookup else dimensions
        try:
            s_dim = sum(dims)
            dim_list = dimensions
        except:
            s_dim = dims
            dim_list = [dimensions]
        scale = np.sqrt(6) / np.sqrt(s_dim)
        return scale * (self.state.rand(*dim_list) * 2 - 1)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def ngrams(seq, k):
    return (seq[i:i+k] for i in range(len(seq) - k + 1))


def contains_subsequence(needle, haystack):
    if len(needle) == 1:
        return needle[0] in haystack
    else:
        return needle in ngrams(haystack, len(needle))


def flatten(lol):
    return [t for l in lol for t in l]


def all_equal(lst):
    return all(x == lst[0] for x in lst)


def split_sequence(sequence, split_element):
    splits = []
    l = []
    for element in sequence:
        if element == split_element:
            if l:
                splits.append(l)
                l = []
        else:
            l.append(element)
    if l:
        splits.append(l)
    return splits


def dump_pickle(obj, fname):
    with open(fname, 'w') as f:
        pickle.dump(obj, f)


def load_pickle(fname):
    with open(fname) as f:
        return pickle.load(f)

def typed_namedtuple(name, fields, type_name="TYPE"):
    # fields should be a list
    assert type_name not in fields
    Inner = namedtuple(name + "_", fields + [type_name])
    class Outer(Inner):
        def __new__(cls, *args, **kwargs):
            kwargs[type_name] = name
            return Inner.__new__(cls, *args, **kwargs)

        def __repr__(self):
            d = self._asdict()
            return "%s(%s)" % (
                name, ", ".join("%s=%s" % (field, d[field])
                                for field in fields)
            )
    return Outer


def run(arg_parser, entry_function):
    arg_parser.add_argument("--pdb", action='store_true')
    arg_parser.add_argument("--ipdb", action='store_true')

    args, other_args = arg_parser.parse_known_args()

    def log(out_file):
        subprocess.call("git rev-parse HEAD", shell=True, stdout=out_file)
        subprocess.call("git --no-pager diff", shell=True, stdout=out_file)
        out_file.write('\n\n')
        out_file.write(' '.join(sys.argv))
        out_file.write('\n\n')
        json.dump(vars(args), out_file, sort_keys=True, indent=2)
        out_file.write('\n\n')

    log(sys.stdout)
    # if 'save_dir' in vars(args) and args.save_dir:
    #     with open(os.path.join(args.save_dir, 'invoke.log'), 'w') as f:
    #         log(f)

    if args.ipdb:
        import ipdb
        ipdb.runcall(entry_function, args)
    elif args.pdb:
        import pdb
        pdb.runcall(entry_function, args)
    else:
        entry_function(args)


def close(x, y, tolerance=1e-3):
    return abs(x - y) < tolerance
