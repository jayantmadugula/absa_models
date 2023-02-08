'''
The classes in this file are responsible for batch loading data during model training.

Each class targets a different data source (and, possibly, usage scenario).
'''

from typing import Iterable, List
from multiprocessing import Pool
import numpy as np
from database_utilities.database_handler import DatabaseHandler
from utilities.data_preparation import split_chunks


class BaseDataLoader():
    def __init__(self):
        raise NotImplementedError('Please use a subclass that inherits from BaseDataLoader.')

    def read(self, idx_range) -> np.ndarray:
        raise NotImplementedError()
    

class InMemoryDataLoader(BaseDataLoader):
    '''
    A simple DataLoader that holds all data entirely in memory.
    '''
    def __init__(self, data: Iterable):
        '''
        `data`: an Iterable that can be sliced by index.
        '''
        self._data = data

    def read(self, idx_range: Iterable[int]) -> Iterable:
        '''
        Returns data from `self._data` in the given `idx_range`.
        '''
        return self._data[idx_range]
        

class MemoryMapDataLoader(BaseDataLoader):
    '''
    Loads data saved to a mmap file.
    '''
    def __init__(self, data_filepath: str):
        ''' 
        `data_filepath`: path to the data mmap file
        '''
        self.data_filepath = data_filepath

    def read(self, idx_range: Iterable[int]) -> np.ndarray:
        '''
        Reads data from `self._data_filepath` in the given `idx_range`.
        '''
        return np.load(self.data_filepath, mmap_mode='r')[idx_range]


class EmbeddedDataLoader(BaseDataLoader):
    '''
    Loads data from a pre-saved `.npy` file, where each filename is equal to the
    index of the data in a larger dataset.
    '''
    def __init__(self, data_path: str, embedding_dim: int, n: int, n_procs: int = 2):
        '''
        Parameters:
        - `data_path`: path to a directory containing `.npy` files; the filenames
        must be each data point's index
        - `embedding_dim`: the data's embedding dimensionality
        - `n`: equivalent to `0.5 * ngram length - 1`
        - `n_procs`: the number of processes to use when loading the data
        '''
        self._dir_path = data_path
        self._emb_dim = embedding_dim
        self._n = n # must be equal to 0.5 * window length - 1
        self._n_procs = n_procs

    def read(self, idx_range) -> np.ndarray:
        '''
        Reads data with indices within the provided `idx_range`.
        '''
        grouped_indices = split_chunks(idx_range, n_procs=self._n_procs)
        
        with Pool(self._n_procs) as p:
            res = p.map(self._fetch_embedded_data, grouped_indices)

        return np.concatenate(res)

    def _fetch_embedded_data(self, indices: Iterable[int]) -> np.ndarray:
        '''
        Fetches data that have an index found in `indices`.
        '''
        np_arrs = []
        for i in indices:
            filename = f'{i}.npy'
            np_arrs.append(np.load(self._dir_path + filename))
        return np.stack(np_arrs)


class SqliteDataLoader(BaseDataLoader):
    '''
    Loads data stored in a SQLite database.
    '''
    def __init__(self, database_path, table_name, data_column_name, vectorizer=None):
        self._db_handler = DatabaseHandler(database_path)
        self._table_name = table_name
        self._data_column_name = data_column_name
        self._vectorizer = vectorizer

    def read(self, idx_range: Iterable[int]) -> np.ndarray | List[str]:
        '''
        Reads data from the connected SQlite3 database.
        '''
        data = self._db_handler.read(
            self._table_name,
            row_indices=idx_range,
            columns=[self._data_column_name]
        )[self._data_column_name].tolist()

        return self._vectorizer(data) if self._vectorizer else data