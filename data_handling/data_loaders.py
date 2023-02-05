'''
The classes in this file are responsible for enabling batch loading
of data from various sources.
'''

from typing import Generator, Iterable, List, Tuple
from multiprocessing import Pool
import numpy as np
from scipy.sparse.csr import csr_matrix
from database_utilities.database_handler import DatabaseHandler


class BaseDataLoader():
    def __init__(self):
        raise NotImplementedError('Cannot initialize an abstract class.')

    def read(self, idx_range) -> np.ndarray:
        raise NotImplementedError()

    def read_metadata(self, idx_range) -> np.ndarray:
        raise NotImplementedError()


class EmbeddedDataLoader(BaseDataLoader):
    '''
    Data Loader used to load data from a pre-saved `.npy` file.

    Currently, this class only supports metadata that is directly
    passed in or saved to an mmap file.
    '''
    def __init__(self, data_path, embedding_dim, n, metadata=None, n_procs=1, is_metadata_copied=True):
        '''
        Parameters:
        - `data_path`: path to a directory containing `.npy` files; the filenames
        must be each data point's index
        - `embedding_dim`: the data's embedding dimensionality
        - `n`: equivalent to `0.5 * ngram length - 1`
        - `metadata`: metadata related to the data being loaded; must be either the 
        actual metadata or the path to an mmap file containing the metadata
        - `n_procs`: the number of processes to use when loading the data
        - `is_metadata_copied`: `True` if `metadata` contains actual metadata, `False`
        if the metadata must be loaded from a mmap file
        '''
        self._dir_path = data_path
        self._emb_dim = embedding_dim
        self._n = n # must be equal to 0.5 * window length - 1
        self._n_procs = n_procs
        self._metadata = metadata # either actual metadata or path to mmap file
        self._is_metadata_copied = is_metadata_copied

    def read(self, idx_range) -> np.ndarray:
        '''
        Reads data with indices within the provided `idx_range`.
        '''
        grouped_indices = self._split_chunks(idx_range)
        
        with Pool(self._n_procs) as p:
            res = p.map(self._fetch_embedded_data, grouped_indices)

        return np.concatenate(res)

    def read_metadata(self, idx_range) -> np.ndarray:
        '''
        Reads metadata in from `self._metadata`.
        '''
        # TODO: Implement ability to read metadata from database as well.
        if self._metadata is None:
            raise ValueError('No metadata provided to initializer.')

        if self._is_metadata_copied:
            selected_metadata = self._metadata[idx_range]
            if type(selected_metadata) is csr_matrix:
                selected_metadata = selected_metadata.toarray()
            return selected_metadata
        else:
            metadata = np.load(self._metadata, mmap_mode='r')
            return metadata[idx_range]

    def _split_chunks(self, idx_range: Iterable[int]) -> Generator[Iterable[int], None, None]:
        '''
        Splits a list of indices into chunks based on `self._n_threads`.

        Yields a list of indices of length <= the calculated chunksize.
        '''
        chunksize = int(len(idx_range) / self._n_procs)
        for i in range(0, len(idx_range), chunksize):
            yield idx_range[i:i + chunksize]

    def _fetch_embedded_data(self, indices: Iterable[int]) -> np.ndarray:
        '''
        Fetches data that have an index found in `indices`.
        '''
        np_arrs = []
        for i in indices:
            filename = f'{i}.npy'
            np_arrs.append(np.load(self._dir_path + filename))
        return np.stack(np_arrs)


class PreSavedDataLoader(BaseDataLoader):
    '''
    Data Loader used to load data stored in an mmap file.

    Supports metadata that is either directly passed in or saved
    to an mmap file.
    '''
    def __init__(self, data_filepath, metadata=None, is_metadata_copied=True):
        '''
        Parameters:
        - `data_filepath`: path to the data mmap file
        - `metadata`: metadata related to the data being loaded; must be either the 
        actual metadata or the path to an mmap file containing the metadata
        - `is_metadata_copied`: `True` if `metadata` contains actual metadata, `False`
        if the metadata must be loaded from a mmap file
        '''
        self._data_filepath = data_filepath
        self._metadata = metadata
        self._is_metadata_copied = is_metadata_copied

    def read(self, idx_range: Iterable[int]) -> np.ndarray:
        '''
        Reads data from `self._data_filepath`.
        '''
        data = np.load(self._data_filepath, mmap_mode='r')
        return data[idx_range]

    def read_metadata(self, idx_range) -> np.ndarray:
        '''
        Reads metadata in from `self._metadata`.
        '''
        if self._metadata is None:
            raise ValueError('No metadata provided to initializer.')

        if self._is_metadata_copied:
            selected_metadata = self._metadata[idx_range]
            if type(selected_metadata) is csr_matrix:
                selected_metadata = selected_metadata.toarray()
            return selected_metadata
        else:
            metadata = np.load(self._metadata, mmap_mode='r')
            return metadata[idx_range]

class SqliteDataLoader(BaseDataLoader):
    '''
    Loads data directly from a Sqlite3 database.
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
    
    def read_metadata(self, idx_range) -> np.ndarray:
        return super().read_metadata(idx_range)

class LabeledDataLoader(BaseDataLoader):
    '''
    Loads data that has corresponding labels.

    Data must be saved to an mmap file. Labels must be directly
    passed in.
    '''
    def __init__(self, data_filepath, labels):
        ''' 
        Parameters:
        - `data_filepath`: path to the data mmap file
        - `labels`: actual labels with indices corresponding to the 
        data pointed at by `data_filepath`
        '''
        self.data_filepath = data_filepath
        self._labels = labels

    def read(self, idx_range: Iterable[int]) -> Tuple[np.ndarray, Iterable]:
        '''
        Reads data from `self._data_filepath` in the given `idx_range`.
        Also returns labels corresponding to the current data batch.
        '''
        data = np.load(self.data_filepath, mmap_mode='r')
        data_batch = data[idx_range]
        label_batch = self._labels[idx_range]
        return (data_batch, label_batch)


class InMemoryDataLoader(BaseDataLoader):
    '''
    Data Loader for data that can be held entirely in memory.

    Expects both data and labels to be directly passed in.
    '''
    def __init__(self, data, labels):
        '''
        Parameters:
        - `data`: iterable of all needed data
        - `labels`: iterable of labels corresponding with entries
        in `data`
        '''
        self.data = data
        self._labels = labels

    def read(self, idx_range: Iterable[int]) -> Tuple[Iterable, Iterable]:
        '''
        Returns data from `self._data` in the given `idx_range`.
        Also returns labels corresponding to the current data batch.
        '''
        data_batch = self.data[idx_range]
        label_batch = self._labels[idx_range]
        return (data_batch, label_batch)