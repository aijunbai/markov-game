# -*- coding: utf-8 -*-



import inspect
import pprint
import random
import sys
from collections import defaultdict

import numpy as np

__author__ = "Aijun Bai"
__copyright__ = "Copyright 2015, Alibaba Inc."
__email__ = "aijunbai@gmail.com"


def makehash():
    return defaultdict(makehash)


def chain_files(file_names):
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                yield line


def drange(start=0.0, stop=1.0, step=0.1):
    eps = 1.0e-6
    r = start
    while r < stop + eps if stop > start else r > stop - eps:
        yield min(max(min(start, stop), r), max(start, stop))
        r += step


def pv(*args, **kwargs):
    for name in args:
        record = inspect.getouterframes(inspect.currentframe())[1]
        frame = record[0]
        val = eval(name, frame.f_globals, frame.f_locals)

        prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
        iostream = sys.stdout if 'stdout' in kwargs and kwargs['stdout'] \
            else sys.stderr

        print('%s%s: %s' % (prefix, name, pprint.pformat(val)), file=iostream)


def weighted_mean(samples, weights):
    return sum(x * w for x, w in zip(samples, weights)) / sum(weights) \
        if sum(weights) > 0.0 else 0.0


def mean(samples):
    return sum(samples) / len(samples) if len(samples) else 0.0


def flatten(x):
    return [y for l in x for y in flatten(l)] if type(x) is list else [x]


def forward(*args):
    print('\t'.join(str(i) for i in args))


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
