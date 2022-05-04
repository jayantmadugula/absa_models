'''
This script will embed a series of documents
using a provided set of pre-trained embeddings.

Global settings for this script are defined in
`parameters.json`.
'''

import argparse
import json
from multiprocessing import Pool
import numpy as np
from pathlib import Path
from functools import partial

from database_utilities.database_handler import DatabaseHandler
from utilities.database_mappings import get_table_name

def validate_embedding_type(emb_type: str):
    if emb_type == 'glove': 
        return

    raise ValueError('Invalid embedding type provided.')

if __name__ == '__main__':
    # Set up script argument parameters.
    parser = argparse.ArgumentParser(description='''
        Creates and saves embedded representations of ngrams,
        documents, or individual words in a corpus.
        
        The raw text data is expected to be saved to a SQLite3 database. Additional settings for this
        scripts can be found in `parameters.json`.
        ''')
    parser.add_argument(
        'dataset',
        nargs=1,
        type=str,
        choices=['restaurantreviews', 'semeval16', 'sst', 'socc'],
        help='identifies which dataset will be used for embedding'
    )
    parser.add_argument(
        '--target',
        '-t',
        nargs=1,
        type=str,
        choices=['ngram', 'document', 'word'],
        default='ngram',
        dest='emb_target',
        help='determines whether the script embeds ngrams, documents, or individual words'
    )
    parser.add_argument(
        '--ngram_len',
        '-n',
        nargs=1,
        type=int,
        default=None,
        dest='ngram_len',
        help='the number of "context" words on either side of the target word in each ngram'
    )
    parser.add_argument(
        '--text_filtering',
        '-f',
        nargs=1,
        type=str,
        choices=['none', 'pos'],
        default='none',
        dest='filtering_option',
        help='identifies whether to use pre-filtered texts for embedding'
    )

    args = parser.parse_args()
    selected_dataset = args.dataset
    embedding_target = parser.emb_target
    ngram_len = parser.ngram_len
    text_filtering_option = parser.filtering_option

    # Get parameters.
    with open('./parameters.json') as params_fp:
        params = json.load(params_fp)

    batch_size = params['ngram_batch_size'] if embedding_target == 'ngram' else params['document_batch_size']
    num_procs = params['num_processes']

    db_path = params['database_path']
    emb_root_path = params['embedding_root_path']
    emb_type = params['embedding_type']

    # Read data from database.
    db_handler = DatabaseHandler(db_path)

    additional_args = 'pos-filter' if text_filtering_option else None
    table_name = get_table_name(
        dataset=selected_dataset,
        get_documents=(embedding_target != 'ngram'),
        get_metadata=False,
        ngram_len=ngram_len,
        args=additional_args
    )
    batched_data = db_handler.read(
        table_name,
        chunksize=batch_size,
        retry=True
    )