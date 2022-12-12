'''
This file contains unit tests for training_utilities/data_preparation.py
'''

import unittest
import numpy as np
import pandas as pd
from string import ascii_lowercase
from utilities import data_preparation as dp

class DataPreparationUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._num_samples = 10
        return super().setUp()

    def test_split_data_randomly(self):
        test_data_df = pd.DataFrame({
            'id': list(range(self._num_samples)),
            'test_text': [ascii_lowercase[i] for i in range(self._num_samples)]
        })
        test_secondary_array = np.arange(self._num_samples)
        train_test_split = 0.2

        expected_train_size = self._num_samples * (1 - train_test_split)
        expected_test_size = self._num_samples * train_test_split

        res = dp.split_data_randomly(
            test_data_df,
            train_test_split,
            test_secondary_array
        )

        assert(len(res) == 4)
        res_train_df = res[0]
        res_test_df = res[1]
        res_train_arr = res[2]
        res_test_arr = res[3]

        assert(res_train_df.shape[0] == expected_train_size)
        assert(res_train_df.shape[1] == test_data_df.shape[1])
        assert(res_train_arr.shape[0] == expected_train_size)

        assert(res_test_df.shape[0] == expected_test_size)
        assert(res_test_df.shape[1] == test_data_df.shape[1])
        assert(res_test_arr.shape[0] == expected_test_size)

        assert((res_train_df['id'].tolist() == res_train_arr).all())
        assert((res_test_df['id'].tolist() == res_test_arr).all())

    def test_calculate_train_test_split(self):
        ratio = 0.2
        train_size, test_size = dp.calculate_train_test_split(
            self._num_samples,
            ratio = ratio
        )
        assert(train_size + test_size == self._num_samples)
        assert(test_size / self._num_samples == ratio)

        ratio = 0.5
        train_size, test_size = dp.calculate_train_test_split(
            self._num_samples,
            ratio = ratio
        )
        assert(train_size + test_size == self._num_samples)
        assert(test_size == train_size)
        assert(test_size / self._num_samples == ratio)

    def test_split_indices_randomly(self):
        train_size = 8

        train_res, test_res = dp.split_indices_randomly(
            self._num_samples,
            train_size
        )

        assert(len(train_res) == train_size)
        assert(len(test_res) == self._num_samples - train_size)

        combined_res = np.concatenate((train_res, test_res))
        assert(len(combined_res) == 10)
        assert(max(combined_res) == self._num_samples - 1)
        assert(min(combined_res) == 0)
        assert((combined_res != np.sort(combined_res)).any())

    def test_split_chunks(self):
        data_list = [0, 1, 2, 3, 4, 5]
        n = 3

        res = list(dp.split_chunks(data_list, n))
        assert(len(res) == 3)
        assert(res[0] == [0, 1])
        assert(res[1] == [2, 3])
        assert(res[2] == [4, 5])
