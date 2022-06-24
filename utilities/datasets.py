from enum import Enum, unique

@unique
class Dataset(Enum):
    '''
    Defines the set of supported datasets.

    Each enum value has some information associated with it.
    '''
    RESTAURANTREVIEWS = 'restaurant_reviews'
    SEMEVAL16 = 'semeval16'
    SST = 'sst'
    SOCC = 'socc'

    @classmethod
    def get_dataset(cls, s: str):
        if s == 'restaurantreviews':
            return Dataset.RESTAURANTREVIEWS
        elif s == 'semeval16':
            return Dataset.SEMEVAL16
        elif s == 'socc':
            return Dataset.SOCC
        elif s == 'sst':
            return Dataset.SST
        else:
            raise ValueError('Invalid dataset option.')

    def get_table_name(
        self, 
        *args,
        get_documents: bool = True, 
        get_metadata: bool = False,
        ngram_len: int = None) -> str:
        # Define a series of nested factory functions for each registered table.
        def construct_restaurantreviews_ngram_table_name(ngram_len: int, *args) -> str:
            base_name = f'restaurantreviews_n={ngram_len}'
            if len(args) == 0: return base_name
            else: return '_'.join([base_name] + list(args))

        def construct_restaurantreviews_document_table_name() -> str:
            return 'restaurantreviews_reviews'

        def construct_restaurantreviews_restaurants_table_name() -> str:
            return 'restaurantreviews_restaurants'

        def construct_semeval16_ngram_table_name(ngram_len: int, *args) -> str:
            base_name = f'semeval16={ngram_len}'
            if len(args) == 0: return base_name
            else: return '_'.join([base_name] + list(args))

        def construct_semeval16_document_table_name() -> str:
            return 'semeval16_reviews'

        def construct_semeval16_metadata_table_name() -> str:
            return 'semeval16_opinion_data'

        def construct_socc_ngram_table_name(ngram_len: int, *args) -> str:
            base_name = f'socc={ngram_len}'
            if len(args) == 0: return base_name
            else: return '_'.join([base_name] + list(args))

        def construct_socc_document_table_name() -> str:
            return 'socc_articles'

        def construct_sst_ngram_table_name(ngram_len: int, *args) -> str:
            base_name = f'sst={ngram_len}'
            if len(args) == 0: return base_name
            else: return '_'.join([base_name] + list(args))

        def construct_sst_document_table_name() -> str:
            return 'sst_phrases'
        
        # Determine the correct factory function to use.
        if self == Dataset.RESTAURANTREVIEWS:
            if get_documents:
                return construct_restaurantreviews_document_table_name()
            elif get_metadata:
                return construct_restaurantreviews_restaurants_table_name()
            elif ngram_len != None:
                return construct_restaurantreviews_ngram_table_name(ngram_len, *args)
            else:
                raise ValueError('Invalid options for the RestaurantReviews dataset.')
        elif self == Dataset.SEMEVAL16:
            if get_documents:
                return construct_semeval16_document_table_name()
            elif get_metadata:
                return construct_semeval16_metadata_table_name()
            elif ngram_len != None:
                return construct_semeval16_ngram_table_name(ngram_len, *args)
            else:
                raise ValueError('Invalid options for the SemEval16 dataset.')
        elif self == Dataset.SOCC:
            if get_documents:
                return construct_socc_document_table_name()
            elif ngram_len != None:
                return construct_socc_ngram_table_name(ngram_len, *args)
            else:
                raise ValueError('Inalid options for the SOCC dataset.')
        elif self == Dataset.SST:
            if get_documents:
                return construct_sst_document_table_name()
            elif ngram_len != None:
                return construct_sst_ngram_table_name(ngram_len, *args)
            else:
                raise ValueError('Invalid options for the SST dataset.')
        else:
            raise ValueError('Invalid dataset option.')

    @staticmethod
    def get_text_column_name(table_name: str):
        ''' Returns the primary text column for each known table. '''
        #TODO Improve this logic to be more extensible.
        if table_name in ['restaurantreviews_reviews', 'semeval16_reviews']:
            return 'review'
        elif table_name == 'socc_articles':
            return 'article_text'
        elif table_name == 'sst_phrases':
            return 'phrase'
        elif '=' in table_name:
            return 'ngram'
