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
from data_handling.embedding_generation import PreTrainedEmbeddings, generate_ngram_matrix

from database_utilities.database_handler import DatabaseHandler
from utilities.data_preparation import split_chunks
from utilities.datasets import Dataset
from utilities.read_write import save_embeddings

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
        type=str,
        choices=['restaurantreviews', 'semeval16', 'sst', 'socc'],
        help='identifies which dataset will be used for embedding'
    )
    parser.add_argument(
        '--target',
        '-t',
        type=str,
        choices=['ngram', 'document', 'word'],
        default='ngram',
        dest='emb_target',
        help='determines whether the script embeds ngrams, documents, or individual words'
    )
    parser.add_argument(
        '--ngram_len',
        '-n',
        type=int,
        default=None,
        dest='ngram_len',
        help='the number of "context" words on either side of the target word in each ngram'
    )
    parser.add_argument(
        '--text_filtering',
        '-f',
        type=str,
        choices=['none', 'pos'],
        default='none',
        dest='filtering_option',
        help='identifies whether to use pre-filtered texts for embedding'
    )
    parser.add_argument(
        '--debug',
        '-d',
        type=bool,
        default=False,
        dest='enable_debug',
        help='passing True for this argument causes the script to skip writing the embedding data'
    )

    args = parser.parse_args()
    selected_dataset = Dataset.get_dataset(args.dataset)
    embedding_target = args.emb_target
    ngram_len = args.ngram_len
    text_filtering_option = args.filtering_option
    debug = args.enable_debug

    # Get parameters.
    with open('./parameters.json') as params_fp:
        params = json.load(params_fp)

    batch_size = params['ngram_batch_size'] if embedding_target == 'ngram' else params['document_batch_size']
    num_procs = params['num_processes']

    db_path = params['database_path']
    emb_root_path = params['embedding_root_path']
    emb_type = PreTrainedEmbeddings.map_string(params['embedding_type'])
    emb_dim = params['embedding_dimension']
    emb_output_dir = params['embedding_output_root_dir']

    # Read data from database.
    db_handler = DatabaseHandler(db_path)

    filter_arg = 'pos-filter' if text_filtering_option else None
    table_name = selected_dataset.get_table_name(
        filter_arg,
        get_documents=(embedding_target != 'ngram'),
        get_metadata=False,
        ngram_len=ngram_len
    )
    batched_data = db_handler.read(
        table_name,
        chunksize=batch_size,
        retry=True
    )
    print(f'Read data from database using {batch_size} element batches from {table_name}.')
    if debug: print(type(batched_data))

    # Embed the data.
    # TODO: This needs to be aware of db table schema information and take parameters into account.
    # Identify the required information for each "target type" in the general case.
    # Then create specific functions to get those columns.

    output_emb_path = f'{emb_output_dir}/{selected_dataset.value}/emb_{table_name}/'
    output_emb_path = output_emb_path.replace('//', '/')

    generate_matrix_partial = partial(
        generate_ngram_matrix,
        embedding_rootpath=emb_root_path,
        embedding_type=emb_type,
        embedding_dimension=emb_dim
    )

    save_embeddings_partial = partial(
        save_embeddings,
        directory_path=output_emb_path,
        debug=debug
    )

    for i, batch in enumerate(batched_data):
        col_name = Dataset.get_text_column_name(table_name)
        batched_doc = split_chunks(batch.loc[:, col_name], num_procs)
        batched_idx = split_chunks(batch.index, num_procs)

        with Pool(num_procs) as p:
            grouped_embeddings = p.map(generate_matrix_partial, batched_doc)
            p.starmap(
                save_embeddings_partial, 
                zip(grouped_embeddings, batched_idx))
    
    print(f'Saved embedding ngrams with size {ngram_len}.')

