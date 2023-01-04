'''
This file contains utility functions for
efficiently loading and saving data necessary
for the scripts in this repository.
'''

from pathlib import Path
from typing import Iterable
import numpy as np


def save_embeddings(
    embeddings_list: Iterable[str], 
    ids_list: Iterable[int], 
    directory_path: str,
    dry_run: bool = False):
    '''
    Saves an embedding matrix to an .npy file.
    '''
    if dry_run:
        print('[Debug] Save function received a list of embeddings' \
            + f' with length {len(embeddings_list)} and a list of ids' \
            + f' with length {len(ids_list)}. The write path is {directory_path}')
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