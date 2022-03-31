'''
This file contains functions useful to prepare data before
training a model.
'''

from typing import Iterable
import pandas as pd
import numpy as np
import math


def split_data_randomly(data_df: pd.DataFrame, train_test_split: float, *args) -> Iterable:
    '''
    Randomly splits any provided data, including `data_df`, into training
    and testing iterables. The relative size of the training and testing
    data is controlled via `train_test_split`.

    Extra data can be provided in `*args` as either `pd.DataFrame` or 
    `np.ndarray` objects. These objects will be split identically to `data_df`
    by index. In other words, the indices are randomly split, but the all
    data sources are split using the same set of training and testing indices.
    
    Parameters:
    - `data_df`: primary data to be split
    - `train_test_split`: train/test size ratio (e.g. 0.2 -> 80/20 train/test split)
    - `args`: only data in `pd.DataFrame` or `np.ndarray` objects are accepted
    '''
    num_samples = data_df.shape[0]
    train_size, _ = calculate_train_test_split(
        num_samples, ratio=train_test_split)
    train_indices, test_indices = split_indices_randomly(
        num_samples, train_size)

    train_df = data_df.iloc[train_indices].reset_index(drop=True)
    test_df = data_df.iloc[test_indices].reset_index(drop=True)

    split_args = [train_df, test_df]
    for arr in args:
        split_args.append(arr[train_indices])
        split_args.append(arr[test_indices])

    return split_args


def calculate_train_test_split(num_samples, ratio=0.2) -> int:
    '''
    Determines the size of the training and testing datasets given
    the number of total samples and the train/test ratio.
    
    Parameters:
    - `num_samples`: the total number of elements for training and testing
    - `ratio`: the train/test ratio (0.2 -> 80/20 train/test split)
    '''
    train_size = math.ceil(num_samples * (1 - ratio))
    test_size = math.floor(num_samples * ratio)
    return train_size, test_size


def split_indices_randomly(num_samples, train_size) -> Iterable[np.ndarray]:
    '''
    Provides two iterables of indices, one defining a training
    dataset and the other defining a testing dataset. The indices
    are randomized before being split.

    The number of indices in the first element of the returned
    arrays is equivalent to `train_size`.

    Parameters:
    - `num_samples`: the total number of elements for training and testing
    - `train_size`: the number of elements in the training dataset
    (the first element returned)
    '''
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    return np.split(indices, [train_size])