'''
These classes interact with the Keras API to provide
data streaming to a model.

For more information, see Keras Sequence documentation.

The subclasses defined here allow for custom behavior
at various points of the training and testing process.

Concrete subclasses should implement:
- __len__()
- on_epoch_end()
- __get_item__()
'''

from typing import Iterable
import numpy as np
from data_handling.data_loaders import BaseDataLoader
from tensorflow.keras.utils import Sequence
import math


class BaseNLPGenerator(Sequence):
    '''
    Implementation of a `keras.Sequence` generator for an ngram-based NLP model.
    This generator pulls a new batch of data at the end of each epoch.
    Separate instances should be created for training, testing, and validation.
    '''
    def __init__(self, data_loader: BaseDataLoader, indices: Iterable, batch_size: int, ngram_len: int, embedding_dim: int):
        '''
        Parameters:
        - `data_loader` must be a class inheriting from `BaseDataLoader`
        - `indices` is an iterable of valid indices
        - `batch_size`: number of rows in each batch
        - `ngram_len`: length of each ngram
        - `embedding_dim`: embedding dimension used on data
        '''
        # Data Generation
        self._data_loader = data_loader
        self._pos_indices = indices if type(indices) == list else list(indices)
        self._neg_indices = indices if type(indices) == list else list(indices)

        # Expected Dimensions
        self._batch_size = batch_size
        self._ngram_len = ngram_len
        self._embedding_dim = embedding_dim

    def __len__(self):
        ''' Number of batches per epoch '''
        return math.ceil(len(self._pos_indices) / self._batch_size)

    def on_epoch_end(self):
        '''
        Responsible for reshuffling the order of the data for the next epoch
        '''
        np.random.shuffle(self._pos_indices)
        np.random.shuffle(self._neg_indices)


class SimpleABAEGenerator(BaseNLPGenerator):
    ''' 
    Subclass of `BaseNLPGenerator` designed to work with
    the SimpleABAE model.

    Two batches of inputs are prepared on each `__getitem__()`
    call, the "positive" and "negative" inputs defined by the
    original authors of ABAE.
    '''
    def __getitem__(self, idx):
        '''
        Responsible for getting a single batch of data.
        '''
        idx_range = range(idx * self._batch_size, (idx + 1) * self._batch_size)
        pos_idx = self._pos_indices[idx_range.start:idx_range.stop]
        neg_idx = self._neg_indices[idx_range.start:idx_range.stop]

        batch_pos = self._data_loader.read(pos_idx)
        batch_neg = self._data_loader.read(neg_idx)

        return [batch_pos, batch_neg], np.ones(batch_pos.shape[0])


class ABAEMetadataGenerator(BaseNLPGenerator):
    ''' 
    Generator class for customized ABAE models that take additional metadata inputs.

    This generator provides the model with an additional metadata batch of data
    for each call to `__get_item__()`. The provided `data_loader` object must implement
    a `.read_metadata()` method for this purpose.
    '''
    def __init__(
        self, 
        metadata_col_name: str, 
        data_loader: BaseDataLoader, 
        indices: Iterable, 
        batch_size: int, 
        ngram_len: int, 
        embedding_dim: int):
        '''
        Parameters:
        - `metadata_col_name` is the column name corresponding to the target metadata in the data source
        - `data_loader` must be a class inheriting from `data_handling.BaseeDataLoader`
        - `indices` is an iterable of valid indices
        - `batch_size`: number of rows in each batch
        - `ngram_len`: length of each ngram
        - `embedding_dim`: embedding dimension used on datae
        '''
        super().__init__(data_loader, indices, batch_size, ngram_len, embedding_dim)
        self._metadata_col_name = metadata_col_name

    def __getitem__(self, idx):
        '''
        Responsible for getting a single batch of data.
        '''
        idx_range = range(idx * self._batch_size, (idx + 1) * self._batch_size)
        pos_idx = self._pos_indices[idx_range.start:idx_range.stop]
        neg_idx = self._neg_indices[idx_range.start:idx_range.stop]

        batch_pos = self._data_loader.read(pos_idx)
        batch_metadata = self._data_loader.read_metadata(self._metadata_col_name, pos_idx)
        batch_neg = self._data_loader.read(neg_idx)

        return [batch_pos, batch_neg, batch_metadata], np.ones(batch_pos.shape[0])


class SentimentGenerator(Sequence):
    def __init__(self, data_loader: BaseDataLoader, indices: Iterable, batch_size: int, embedding_dim: int):
        self._data_loader = data_loader
        self._indices = indices
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim

    def __len__(self):
        return math.ceil(len(self._indices) / self._batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self._indices)

    def __getitem__(self, idx):
        idx_range = range(idx * self._batch_size, (idx + 1) * self._batch_size)
        batch_idx = self._indices[idx_range.start:idx_range.stop]
        train_batch, test_batch = self._data_loader.read(batch_idx)
        return train_batch, test_batch