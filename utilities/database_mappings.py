'''
This file defines mappings from various 
variables and parameters to database
tables, columns, and indices.
'''

def get_table_name(
    dataset: str, 
    *args,
    get_documents: bool = True, 
    get_metadata: bool = False,
    ngram_len: int = None) -> str:

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
    
    if dataset == 'restaurantreviews':
        if get_documents:
            return construct_restaurantreviews_document_table_name()
        elif get_metadata:
            return construct_restaurantreviews_restaurants_table_name()
        elif ngram_len != None:
            return construct_restaurantreviews_ngram_table_name(ngram_len, *args)
        else:
            raise ValueError('Invalid options for the RestaurantReviews dataset.')
    elif dataset == 'semeval16':
        if get_documents:
            return construct_semeval16_document_table_name()
        elif get_metadata:
            return construct_semeval16_metadata_table_name()
        elif ngram_len != None:
            return construct_semeval16_ngram_table_name(ngram_len, *args)
        else:
            raise ValueError('Invalid options for the SemEval16 dataset.')
    elif dataset == 'socc':
        if get_documents:
            return construct_socc_document_table_name()
        elif ngram_len != None:
            return construct_socc_ngram_table_name(ngram_len, *args)
        else:
            raise ValueError('Inalid options for the SOCC dataset.')
    elif dataset == 'sst':
        if get_documents:
            return construct_sst_document_table_name()
        elif ngram_len != None:
            return construct_sst_ngram_table_name(ngram_len, *args)
        else:
            raise ValueError('Invalid options for the SST dataset.')
    else:
        raise ValueError('Invalid dataset option.')


def get_text_column_name(table_name: str):
    ''' Returns the primary text column for each known table. '''
    if table_name in ['restaurantreviews_reviews', 'semeval16_reviews']:
        return 'review'
    elif table_name == 'socc_articles':
        return 'article_text'
    elif table_name == 'sst_phrases':
        return 'phrase'
    elif '=' in table_name:
        return 'ngram'