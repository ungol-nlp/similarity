# -*- coding: utf-8 -*-


from ungol.wmd import wmd

import attr
import numpy as np
from tabulate import tabulate

from typing import Any
from typing import Set
from typing import List
from typing import Tuple


@attr.s
class ScoreData:
    """

    Saves the raw data used for calculating a score.
    This classes' data is configured from outside to be adjustable.

    """

    name:    str = attr.ib()
    score: float = attr.ib()

    docs: Tuple[wmd.Doc] = attr.ib()

    common_unknown: Set[str] = attr.ib()

    # ---

    global_rows = attr.ib(default=attr.Factory(list))

    local_rows = attr.ib(default=attr.Factory(list))
    local_columns = attr.ib(default=attr.Factory(list))

    # ---

    def __attr_post_init__(self):
        assert(len(self.tokens))

    def _unpack(self, ds: List[Tuple[str, Tuple[Any, Any]]]):
        headers, columns = zip(*ds)
        row1, row2 = zip(*columns)
        return headers, row1, row2

    def _draw_border(self, s: str) -> str:
        lines = s.splitlines()
        maxl = max(len(l) for l in lines)

        # clock-wise, starting north (west)
        borders = '-', '  |', '-', '|  '
        edges = '+--', '--+', '--+', '+--'

        first = f'{edges[0]}' + f'{borders[0]}' * maxl + f'{edges[1]}'
        midfmt = f'{borders[3]}%-' + str(maxl) + f's{borders[1]}'
        last = f'{edges[3]}' + f'{borders[2]}' * maxl + f'{edges[2]}'

        mid = [midfmt % s for s in lines]
        return '\n'.join([first] + mid + [last])

    # ---

    def _str_global_table(self, buf: List[str]) -> None:
        buf.append(f'\n{self.name} score : {self.score}\n')

        if len(self.global_rows):
            buf.append(tabulate(self.global_rows))
            buf.append('')

    def _str_local_rows(self, buf: List[str], a: int, b: int) -> None:
        name1, name2 = self.docs[a].name, self.docs[b].name
        buf.append(f'\ncomparing: "{name1}" to "{name2}"\n')

        if len(self.local_rows):
            rows = ((n, t[a]) for n, t in self.local_rows)
            buf.append(tabulate(rows))
            buf.append('')

        buf.append('')

    def _str_local_columns(self, buf: List[str], a: int) -> None:
        headers, *rows = self._unpack(self.local_columns)
        rows = list(zip(*rows[a]))

        headers = ('no', ) + headers
        rows.sort(key=lambda t: t[-1], reverse=True)

        sims_table = tabulate(
            rows, headers=headers,
            showindex='always', floatfmt=".4f")

        buf.append(sims_table)

    def _str_common_unknown(self, buf: List[str], a: int) -> None:
        buf.append('\n')
        if not len(self.common_unknown):
            buf.append('Common unknown words: None\n')
            return

        buf.append('Common unknown words:\n')
        headers = ('token', 'count')
        tab_data = [(tok, self.docs[a].unknown[tok])
                    for tok in self.common_unknown]

        unknown_table = tabulate(tab_data, headers=headers)
        buf.append(unknown_table)

    # ---

    def __str__(self) -> str:
        buf = []

        self._str_global_table(buf)
        buf.append(self.docstr(first=True) + '\n')
        buf.append(self.docstr(first=False) + '\n')

        return '\n'.join(buf)

    # ---

    def docstr(self, first: bool = True) -> str:
        a, b = (0, 1) if first else (1, 0)

        buf = []

        self._str_local_rows(buf, a, b)
        self._str_local_columns(buf, a)

        buf += ['', '']

        return self._draw_border('\n'.join(buf))

    def add_local_column(
            self, name: str,
            d1_data: np.array, d2_data:
            np.array) -> None:

        assert len(d1_data.shape) == 1 and len(d2_data.shape) == 1
        assert d1_data.shape[0] == len(self.docs[0])
        assert d2_data.shape[0] == len(self.docs[1])

        self.local_columns.append((name, (d1_data, d2_data)))

    def add_local_row(self, name: str, d1_data: Any, d2_data: Any):
        self.local_rows.append((name, (d1_data, d2_data)))

    def add_global_row(self, name: str, data: Any):
        self.global_rows.append((name, data))
