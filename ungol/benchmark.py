#!/usr/bin/env python
# -*- coding: utf-8 -*-

import timeit


# Use the min() rather than the average of the timings. That is a
# recommendation from me, from Tim Peters, and from Guido van
# Rossum. The fastest time represents the best an algorithm can
# perform when the caches are loaded and the system isn't busy with
# other tasks. All the timings are noisy -- the fastest time is the
# least noisy. It is easy to show that the fastest timings are the
# most reproducible and therefore the most useful when timing two
# different implementations. – Raymond Hettinger Apr 3 '12


setup = '''
from ungol import wmd
import numpy as np

bytecount = 32
code1 = (np.random.randn(bytecount) * 255).astype(dtype=np.uint8)
code2 = (np.random.randn(bytecount) * 255).astype(dtype=np.uint8)
'''


def main():
    print('\n', 'welcome to vngol wmd benchmarks'.upper(), '\n')

    repetitions = (20, 1000000)

    t_bincount = min(timeit.Timer(
        'wmd.hamming_bincount', setup=setup
    ).repeat(*repetitions))
    print('bincount time', t_bincount)

    t_bitmask = min(timeit.Timer(
        'wmd.hamming_bitmask', setup=setup
    ).repeat(*repetitions))
    print('bitmask time:', t_bitmask)


if __name__ == '__main__':
    main()
