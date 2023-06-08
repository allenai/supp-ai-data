from typing import Iterable, Iterator, Tuple, List
import itertools
import numpy as np


def flatten(l: List) -> List:
    return [item for sublist in l for item in sublist]

# break a list l into n-sized chunks
def chunk_iter(iterable: Iterable, batch_size: int) -> Tuple:
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield chunk


# break a list l into n chunks
def make_chunks(l: List, n: int) -> Iterator[List]:
    chunk_size = int(np.ceil(len(l) / n))
    for i in range(n):
        yield l[i * chunk_size: (i+1) * chunk_size]