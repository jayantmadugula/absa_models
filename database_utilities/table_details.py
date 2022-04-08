'''
This file contains the classes defining the details
of how each dataset is stored in the SQLite3 database.
'''


class GenericTableDetails():
    '''
    Defines an abstract table details class.
    '''
    def __init__(self) -> None:
        raise NotImplementedError('Please use a concrete class.')

    def build_table_name(self) -> str:
        raise NotImplementedError('Please use a concrete class.')


class RestaurantReviewsNgramTableDetails(GenericTableDetails):
    def __init__(self, ngram_size: int, include_pos_filtering: bool) -> None:
        self._n = ngram_size
        self._pos_filtering = include_pos_filtering

    def build_table_name(self) -> str:
        base_name = f'restaurantreviews_n={self._n}'
        if self._pos_filtering: return base_name + '_pos_filter'
        else: return base_name


class Semeval16NgramTableDetails(GenericTableDetails):
    def __init__(self, ngram_size: int, include_pos_filtering: bool) -> None:
        self._n = ngram_size
        self._pos_filtering = include_pos_filtering

    def build_table_name(self) -> str:
        base_name = f'semeval16={self._n}'
        if self._pos_filtering: return base_name + '_pos_filter'
        else: return base_name


class SOCCNgramTableDetails(GenericTableDetails):
    def __init__(self, ngram_size: int, include_pos_filtering: bool) -> None:
        self._n = ngram_size
        self._pos_filtering = include_pos_filtering

    def build_table_name(self) -> str:
        base_name = f'socc={self._n}'
        if self._pos_filtering: return base_name + '_pos_filter'
        else: return base_name


class SSTNgramTableDetails(GenericTableDetails):
    def __init__(self, ngram_size: int, include_pos_filtering: bool) -> None:
        self._n = ngram_size
        self._pos_filtering = include_pos_filtering

    def build_table_name(self) -> str:
        base_name = f'sst={self._n}'
        if self._pos_filtering: return base_name + '_pos_filter'
        else: return base_name