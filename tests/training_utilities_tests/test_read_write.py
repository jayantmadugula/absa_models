'''
This file contains unit tests for training_utilities/data_preparation.py
'''

import unittest
from utilities.read_write import save_embeddings
import numpy as np
import os
import shutil

class ReadWriteUnitTests(unittest.TestCase):
    temp_dirpath = './tests/temp/'

    def setUp(self) -> None:
        os.mkdir(ReadWriteUnitTests.temp_dirpath)
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(ReadWriteUnitTests.temp_dirpath)
        return super().tearDown()

    def test_save_embeddings(self):
        test_embeddings = [np.random.random(5) for _ in range(0, 10)]
        test_ids = [i for i in range(0, 10)]

        save_embeddings(
            test_embeddings, 
            test_ids, 
            ReadWriteUnitTests.temp_dirpath)

        dir_contents = os.listdir(ReadWriteUnitTests.temp_dirpath)
        print(dir_contents)
        assert(len(dir_contents) == 10)
        assert(sorted(dir_contents) == sorted([f'{i}.npy' for i in range(0, 10)]))
