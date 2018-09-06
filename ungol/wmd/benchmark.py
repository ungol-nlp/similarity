#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tabulate import tabulate

import os
import timeit
import argparse
import multiprocessing as mp
from datetime import datetime

from typing import List
from typing import Tuple
from typing import Callable


# timeit options
REPS = (7, 10000)

# ---


def ts(info: str) -> Callable[[None], None]:
    pid = os.getpid()

    tstart = datetime.now()
    print('[{}] running: {}'.format(pid, info))

    def _done():
        tdelta = (datetime.now() - tstart).total_seconds() * 1000
        print('[{pid}] {info} took: {:.3f}ms'.format(
            tdelta, info=info, pid=pid))

    return _done


def _timeit(cmd: str, setup: str) -> float:
    """
    Returns the best runs time in miliseconds
    """
    agg_time = min(timeit.Timer(cmd, setup=setup).repeat(*REPS))
    return agg_time / REPS[1] * 1000


def _run(info: str, cmd: str, setup: str):
    done = ts(info)
    res = _timeit(cmd, setup)
    done()

    return info, res


def _run_worker(args):
    return _run(*args)


def _benchmark(title: str, tasks: List[Tuple[str]]):

    fmt = 'running {} benchmark ({} times, {} iterations)'
    print(fmt.format(title, *REPS))

    assert all(len(args) == 3 for args in tasks)
    with mp.Pool() as pool:
        tab_data = pool.map(_run_worker, tasks)

    # tab_data = []
    # for args in tasks:
    #     tab_data.append(_run(*args))

    tab_data.sort(key=lambda t: t[1])
    print('\n', tabulate(tab_data, headers=('name', 'time (ms)')), '\n')


def _debug_setup(setup: str):
    eval(setup)
    import pdb
    pdb.set_trace()


# ---


def hamming_distance():

    setup = 'from ungol.wmd import wmd\n'
    setup += 'code1, code2 = wmd.mock_codes(32)'

    _benchmark('hamming distance', [

        ('binary counting with bin()',
         'wmd.hamming_bincount(code1, code2)',
         setup, ),

        ('bitshifts with vanilla python',
         'wmd.hamming_bitmask(code1, code2)',
         setup, ),

        ('lookup table',
         'wmd.hamming_lookup(code1, code2)',
         setup, ),

    ])


# ---


def distance_matrix():

    def gen_setup(*sizes: int):
        for n in sizes:
            setup = 'from ungol.wmd import wmd\n'
            setup += 'doc = wmd.mock_doc({n})'.format(n=n)
            yield n, setup

    benchmarks = []
    for n, setup in gen_setup(10, 20, 30, 40, 50, 100):
        benchmarks += [

            ('naive loop ({n} x {n})'.format(n=n),
             'wmd.distance_matrix_loop(doc, doc)',
             setup, ),

            ('vectorized numpy ({n} x {n})'.format(n=n),
             'wmd.distance_matrix_vectorized(doc, doc)',
             setup, ),

            ('numpy with lookup table ({n} x {n})'.format(n=n),
             'wmd.distance_matrix_lookup(doc, doc)',
             setup, ),

        ]

    for n, setup in gen_setup(500, 1000, 2000):
        benchmarks += [

            ('numpy with lookup table ({n} x {n})'.format(n=n),
             'wmd.distance_matrix_lookup(doc, doc)',
             setup, ),

        ]

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
