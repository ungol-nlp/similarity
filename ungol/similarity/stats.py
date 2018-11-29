# -*- coding: utf-8 -*-


from ungol.index import index as uii

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

    docs: Tuple[uii.Doc] = attr.ib()

    common_unknown: Set[str] = attr.ib(default=attr.Factory(set))

    # ---

    global_rows = attr.ib(default=attr.Factory(list))
    global_columns = attr.ib(default=attr.Factory(list))

    local_rows = attr.ib(default=attr.Factory(list))
    local_columns = attr.ib(default=attr.Factory(list))

    # ---

    def __attr_post_init__(self):
        assert(len(self.tokens))

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

    def _str_rows(self, src, buf):
        buf.append(tabulate(src))
        buf.append('')

    def _str_global_rows(self, buf: List[str]) -> None:
        buf.append(f'\n{self.name} score : {self.score}\n')
        if len(self.global_rows):
            self._str_rows(self.global_rows, buf)

    def _str_local_rows(self, buf: List[str], a: int, b: int) -> None:
        name1, name2 = self.docs[a].name, self.docs[b].name
        buf.append(f'\ncomparing: "{name1}" to "{name2}"\n')

        if len(self.local_rows):
            rows = ((n, t[a]) for n, t in self.local_rows)
            self._str_rows(rows, buf)

        buf.append('')

    # ---

    def _str_cols(self, headers, rows, buf):
        headers = ('no', ) + headers
        rows.sort(key=lambda t: t[-1], reverse=True)

        table = tabulate(
            rows, headers=headers,
            showindex='always', floatfmt=".4f")

        buf.append(table)

    def _str_local_columns(self, buf: List[str], a: int) -> None:
        if not len(self.local_columns):
            return

        headers, columns = zip(*self.local_columns)

        # extract the a-th data column
        # note: a = zip(*zip(*a))
        rows = list(zip(*list(zip(*columns))[a]))
        self._str_cols(headers, rows, buf)

    def _str_global_columns(self, buf: List[str]) -> None:
        if not len(self.global_columns):
            return

        headers, columns = zip(*self.global_columns)
        rows = list(zip(*columns))
        self._str_cols(headers, rows, buf)

    # ---

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

        self._str_global_rows(buf)
        buf.append('')

        self._str_global_columns(buf)
        buf.append('')

        buf.append(self.docstr(first=True) + '\n')
        buf.append(self.docstr(first=False) + '\n')

        return '\n'.join(buf)

    # ---

    def docstr(self, first: bool = True) -> str:
        a, b = (0, 1) if first else (1, 0)

        if not len(self.local_rows) and not len(self.local_columns):
            return f'No data for {self.docs[a].name}'

        buf = []

        self._str_local_rows(buf, a, b)
        self._str_local_columns(buf, a)

        buf += ['', '']

        return self._draw_border('\n'.join(buf))

    # ---

    def _add_column(self, coll, name, data):
        if len(coll):
            len_data, len_coll = len(data), len(coll[0][1])
            assert len_data == len_coll, f'{len_data} =/= {len_coll} ({name})'

        coll.append((name, data))

    def add_local_column(self, name: str, d1: np.array, d2: np.array):
        self._add_column(self.local_columns, name, (d1, d2))

    def add_global_column(self, name: str, data: np.array):
        self._add_column(self.global_columns, name, data)

    def _add_row(self, coll, name, data):
        if len(coll):
            assert len(data[0]) == len(coll[0][1][0])
            assert len(data[1]) == len(coll[0][1][1])

        coll.append((name, data))

    def add_local_row(self, name: str, d1_data: Any, d2_data: Any):
        self._add_row(self.local_rows, name, (d1_data, d2_data))

    def add_global_row(self, name: str, data: Any):
        self._add_row(self.global_rows, name, data)
