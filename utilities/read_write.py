'''
This file contains utility functions for
efficiently loading and saving data necessary
for the scripts in this repository.
'''

from pathlib import Path
from typing import Iterable
import numpy as np


def save_embeddings(
    embeddings_list: Iterable[np.ndarray], 
    ids_list: Iterable[int], 
    directory_path: str,
    dry_run: bool = False):
    '''
    Saves an embedding matrix to an .npy file.

    Parameters:
    - `embeddings_list`: iterable of embeddings (1-dimensional arrays of floats)
    - `ids_list`: iterable of unique ids for each element in `embeddings_list`, with the same ordering
    - `directory_path`: path to the directory where the embeddings will be saved
    - `dry_run`: if True, no embeddings will actually be saved, but additional logging is provided

    `embeddings_list` and `ids_list` must have the same number of elements and ordering (i.e. `ids_list`[2] must be the unique id for `embeddings_list[2]`).
    '''
    if dry_run:
        print('[Debug] Save function received a list of embeddings' \
            + f' with length {len(embeddings_list)} and a list of ids' \
            + f' with length {len(ids_list)}. The write path is {directory_path}')

        print(f'[Debug] Saving data with starting index: {list(ids_list)[0]}')
        return

    try:
        p = Path(directory_path)
        p.mkdir(parents=True)
    except FileExistsError:
        # If the directory already exists, continue!
        pass

    for ngram_id, embedding in zip(ids_list, embeddings_list):
        filename = directory_path + '{}'.format(ngram_id)
        np.save(filename, embedding)