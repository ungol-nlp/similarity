#!/usr/bin/env python
# -*- coding: utf-8 -*-

import timeit
import argparse


# Use the min() rather than the average of the timings. That is a
# recommendation from me, from Tim Peters, and from Guido van
# Rossum. The fastest time represents the best an algorithm can
# perform when the caches are loaded and the system isn't busy with
# other tasks. All the timings are noisy -- the fastest time is the
# least noisy. It is easy to show that the fastest timings are the
# most reproducible and therefore the most useful when timing two
# different implementations. – Raymond Hettinger Apr 3 '12


setup = '''
from ungol.wmd import wmd
import numpy as np

bytecount = 32
code1 = (np.random.randn(bytecount) * 255).astype(dtype=np.uint8)
code2 = (np.random.randn(bytecount) * 255).astype(dtype=np.uint8)
'''


def main():
    print('\n', 'welcome to vngol wmd benchmarks'.upper(), '\n')

    repetitions = (7, 10000)

    t_bincount = min(timeit.Timer(
        'wmd.hamming_bincount(code1, code2)', setup=setup
    ).repeat(*repetitions))
    print('bincount time', t_bincount)

    t_bitmask = min(timeit.Timer(
        'wmd.hamming_bitmask(code1, code2)', setup=setup
    ).repeat(*repetitions))
    print('bitmask time:', t_bitmask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--all', action='store_true',
        help='run all tests')

    parser.add_argument(
        '--hamming-distance', action='store_true',
        help='run tests for hamming distance calculations')

    return parser.parse_args()


if __name__ == '__main__':
    main()
