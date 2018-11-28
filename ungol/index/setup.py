# -*- coding: utf-8 -*-

from ungol.common import logger
from ungol.index import index as uii

import attr
import nltk
from tqdm import tqdm as _tqdm

import re
import os
import sys
import enum
import functools
import multiprocessing as mp

from typing import Tuple
from typing import Generator


log = logger.get('index.setup')
tqdm = functools.partial(_tqdm, ncols=80)


# ---

sys.path.append('lib/CharSplit')
import char_split  # noqa

# ---


#  multiprocessing infrastructure:
#
#  Topology:
#                          rq                        wq
#  reader (main process) 1 -> n [transform tokens] n -> 1 receiver
#           _reader()               _processor()            _receiver()
#
#  Protocol:
#
#  reader sends data to processors
#  reader sends None if done
#  processors send (id, ...) to the writer
#  processors send (id, None) when done
#  if all registered id's sent None, receivers generator exits

# interprocess communication - initialized by the
# Indexer context management
rq, wq = None, None


# class Protocol(enum.Enum):

#     DONE = enum.auto()
#     KILL = enum.auto()


# @attr.s
# class Context:

#     worker: Tuple[int] = attr.ib()


Mapping = Tuple[str, Tuple[str]]


# accepts data generated by _read
# returns doc_id, tokens
_tok_words = functools.partial(nltk.tokenize.word_tokenize, language='german')
_split_word = char_split.split_compound

_re_inc = re.compile(r'\s')
_re_sep = re.compile(r'[/\()\-\[\]]')


def _clean_text(text: str) -> str:
    text = text.lower()
    text = _re_sep.sub(' ', text)
    text = ''.join(e for e in text if e.isalnum() or _re_inc.match(e))
    return text


def _process(data: Tuple[str, str]) -> Mapping:
    doc_id, text = data

    tokens = []

    clean_text = _clean_text(text)
    tokenized = _tok_words(clean_text)

    for token in tokenized:
        token = token.strip()
        tokens.append(token)

        compounds = [token]
        while len(compounds):
            candidate = compounds.pop()
            confidence, *words = _split_word(candidate)[0]

            if confidence > 0.6:
                words = [w.lower() for w in words]
                tokens += words
                compounds += words

    return doc_id, tokens


def _processor(wid: int):
    # reads rq, writes to wq
    pid = os.getpid()
    log.info(f'[{pid}] ({wid}) raised')

    progress = tqdm(position=wid+3, desc=f'worker [{wid}]')

    count = 0
    while True:
        data = rq.get()

        if data is None:
            log.info(f'[{pid}] ({wid}) received death pill')
            wq.put((wid, None))
            break

        else:
            toks = _process(data)
            wq.put((wid, toks))
            count += 1
            progress.update(1)

    log.info(f'[{pid}] ({wid}) dies (processed {count})')
    progress.close()


def _receiver(fn, expected):
    # reads wq
    pid = os.getpid()
    log.info(f'[{pid}] receiver raised')

    def proxy(expected):
        registered = set(expected)
        while len(registered):

            wid, content = wq.get()
            assert wid in registered

            if content is not None:
                yield content

            else:
                registered.remove(wid)
                log.info(f'[{pid}] ({wid}) signaled death')

    res = fn(proxy(expected))
    log.info(f'[{pid}] receiver dies')
    return res


def _read(gen: Generator[str, None, None], processes: int, total: int = None):
    pid = os.getpid()
    log.info(f'[{pid}] reader process spawned')

    for text in tqdm(gen(), total=total, desc='read', position=1):
        rq.put(text)

    for _ in range(processes):
        rq.put(None)

    log.info(f'[{pid}] reader process dies')


@attr.s
class Indexer:
    """

    Indexing is **NOT** thread-safe. Only one indexing procedure may
    run at a time. If ever necessary, just make sure to not use the
    global queues for interproc comm.

    The iterator passed to the context object is initialized in a
    seperate process. Make sure that it does not depend on any
    non-primitive data of the main process.

    """

    index:   uii.Index = attr.ib()
    processes:     int = attr.ib()

    def _insert(self,
                gen: Generator[Tuple[str, str], None, None],
                total: int = None) -> int:
        """

        This function blocks until all data provided by
        the generator was processed and added to the index.

        """
        assert rq is not None and wq is not None
        assert len(self._wids)

        pid = os.getpid()
        log.info(f'[{pid}] (main) creating reader process')

        p_reader = mp.Process(
            target=_read,
            args=(gen, self.processes, ),
            kwargs={'total': total})

        p_reader.start()
        progress = tqdm(desc='indexed', total=total)

        log.info(f'[{pid}] (main) awaiting data to index')
        busy = self._wids.copy()

        # spawn

        for wid in busy:
            log.info(f'[{pid}] (main) dispatching process {wid}')
            mp.Process(target=_processor, args=(wid, )).start()

        # work

        count = 0
        while len(busy):
            wid, content = wq.get()

            assert wid in busy
            if content is None:
                log.info(f'[{pid}] (main) worker ({wid}) signaled death')
                busy.remove(wid)
                continue

            doc_id, tokens = content

            try:
                doc = uii.Doc.from_tokens(doc_id, tokens, self.index.ref)
                self.index += doc
                count += 1
                progress.update(1)

            except uii.DocumentEmptyException:
                self.index.ref.skipped.append(doc_id)

        progress.close()
        print('\n' * (self.processes + 2))
        print('-' * 80)
        print()

        # reader might be the slowest component
        log.info('(main) awaiting reader to finish')
        p_reader.join()
        log.info('(main) reader finished, returning')

        return count

    def __enter__(self):
        global rq
        global wq
        assert rq is None and wq is None

        pid = os.getpid()
        log.info(f'[{pid}] (main) initializing {self.processes} worker')

        rq, wq = mp.Queue(), mp.Queue()
        self._wids = set(range(self.processes))
        return self._insert

    def __exit__(self, *args):
        global rq
        global wq
        assert rq is not None and wq is not None
        pid = os.getpid()

        rq, wq = None, None
        log.info(f'[{pid}] (main) receiver finished; exiting context')
