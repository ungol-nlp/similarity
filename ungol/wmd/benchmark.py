#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tabulate import tabulate

import timeit
import argparse
from datetime import datetime

from typing import List
from typing import Tuple
from typing import Callable


# Use the min() rather than the average of the timings. That is a
# recommendation from me, from Tim Peters, and from Guido van
# Rossum. The fastest time represents the best an algorithm can
# perform when the caches are loaded and the system isn't busy with
# other tasks. All the timings are noisy -- the fastest time is the
# least noisy. It is easy to show that the fastest timings are the
# most reproducible and therefore the most useful when timing two
# different implementations. – Raymond Hettinger Apr 3 '12


# timeit options
REPS = (5, 1000)

# ---


def ts(info: str) -> Callable[[None], None]:
    tstart = datetime.now()
    print('running: {} - '.format(info), end='')

    def _done(*args, **kwargs):
        tdelta = (datetime.now() - tstart).total_seconds()
        print('took: {delta}ms'.format(*args, delta=tdelta, **kwargs))

    return _done


def _timeit(cmd: str, setup: str):
    return min(timeit.Timer(cmd, setup=setup).repeat(*REPS))


def _run(info: str, cmd: str, setup: str):
    done = ts(info)
    res = _timeit(cmd, setup)
    done()

    return info, res


def _benchmark(title: str, tasks: List[Tuple[str]]):

    fmt = 'running {} benchmark ({} times, {} iterations)'
    print(fmt.format(title, *REPS))
    tab_data = []

    for args in tasks:
        tab_data.append(_run(*args))

    tab_data.sort(key=lambda t: t[1])
    print('\n', tabulate(tab_data, headers=('name', 'time')), '\n')


def _debug_setup(setup: str):
    eval(setup)
    import pdb
    pdb.set_trace()


# ---


def hamming_distance():

    setup = '''
from ungol.wmd import wmd
import numpy as np

bytecount = 32
code1 = (np.random.randn(bytecount) * 255).astype(dtype=np.uint8)
code2 = (np.random.randn(bytecount) * 255).astype(dtype=np.uint8)
'''

    _benchmark('hamming distance', [

        ('binary counting with bin()',
         'wmd.hamming_bincount(code1, code2)',
         setup, ),

        ('bitshifts with vanilla python',
         'wmd.hamming_bitmask(code1, code2)',
         setup, )

    ])


# ---


def distance_matrix():

    init = '''
from ungol.wmd import wmd
import numpy as np

byte_c = 32
n_max = 10000

tokens = ['nope' for _ in range(n_max)]
codes = (np.random.randn(n_max, byte_c) * 255).astype(dtype=np.uint8)
dists = (np.random.randn(n_max, 4) * 255).astype(dtype=np.uint8)
'''

    benchmarks = []
    for n in 10, 20, 30, 40, 50:
        doc_init = '''
doc = wmd.Doc(
    tokens=tokens[:{n}],
    codes=codes[:{n}],
    dists=dists[:{n}], )
'''.format(n=n)

        setup = init + doc_init
        benchmarks.append(

            ('naive loop ({n} x {n})'.format(n=n),
             'wmd.distance_matrix_loop(doc, doc)',
             setup, ),

            ('vectorized numpy ({n} x {n})'.format(n=n),
             'wmd.distance_matrix_vectorized(doc, doc)',
             setup, ),

        )

    _benchmark('distance matrix', benchmarks)


def main():
    print('\n', 'welcome to vngol wmd benchmarks'.upper(), '\n')

    global REPS
    mapping = (
        'hamming distance',
        'distance matrix', )

    benchmarks = {'--' + key.replace(' ', '-'): key.replace(' ', '_')
                  for key in mapping}

    parser = argparse.ArgumentParser()

    for option in benchmarks:
        parser.add_argument(option, action='store_true')

    # ---

    parser.add_argument(
        '--run-all', action='store_true',
        help='run all tests', )

    parser.add_argument(
        '--repeat', nargs=1, type=int, default=[REPS[0]],
        help='how many times to call timeit', )

    parser.add_argument(
        '--number', nargs=1, type=int, default=[REPS[1]],
        help='number argument for timeit()'
    )

    # ---

    args = parser.parse_args()
    dict_args = vars(args)

    REPS = args.repeat[0], args.number[0]

    if args.run_all:
        print('running all benchmarks')
        for benchmark in benchmarks.values():
            globals()[benchmark]()

    else:
        if not any([dict_args[k] for k in benchmarks.values()]):
            print('no benchmark selected...')
            return

        for benchmark in benchmarks.values():
            if dict_args[benchmark]:
                globals()[benchmark]()


if __name__ == '__main__':
    main()
