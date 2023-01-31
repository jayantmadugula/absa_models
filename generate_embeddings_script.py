'''
This script will embed a series of documents
using a provided set of pre-trained embeddings.

Global settings for this script are defined in
`parameters.json`.
'''

import argparse
import json
from multiprocessing import Pool
from functools import partial
from data_handling.document_tokenizers import simple_tokenizer
from data_handling.embedding_generation import PreTrainedEmbeddings, generate_ngram_matrix

from database_utilities.database_handler import DatabaseHandler
from utilities.data_preparation import split_chunks
from utilities.read_write import save_embeddings

def validate_embedding_type(emb_type: str):
    if emb_type == 'glove': 
        return

    raise ValueError('Invalid embedding type provided.')

def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='''
        Creates and saves embedded representations of ngrams,
        documents, or individual words in a corpus.
        
        The raw text data is expected to be saved to a SQLite3 database. Additional settings for this
        scripts can be found in `parameters.json`.
        ''')
    parser.add_argument(
        'table_name',
        type=str,
        help='Identifies the name of the database table to use for embedding.'
    )
    parser.add_argument(
        'data_column_name',
        type=str,
        help='Identifies the name of the column with text to be embedded.'
    )
    parser.add_argument(
        '--target_type',
        '-t',
        type=str,
        choices=['ngram', 'document', 'word'],
        default='ngram',
        dest='emb_target',
        required=True,
        help='Determines whether the script embeds ngrams, documents, or individual words.'
    )
    parser.add_argument(
        '--output_directory_name',
        '-o',
        type=str,
        dest='output_dir_name',
        help='''
        If provided, the script places the embeddings into this directory; the resulting path
        would be: ./<output_directory_name>/<table_name>/emb_<table_name>/ngram_idx.npy

        If not provided, the script will use the table name for the parent directory's name.
        '''
    )
    parser.add_argument(
        '--text_filtering',
        '-f',
        type=str,
        choices=['none', 'pos'],
        default='none',
        dest='filtering_option',
        help='Identifies whether to use pre-filtered texts for embedding.'
    )
    parser.add_argument(
        '--debug',
        '-d',
        type=bool,
        default=False,
        dest='enable_debug',
        help='When enabled, the script skips writing the embedding data to disk and prints debug information.'
    )

    return parser

if __name__ == '__main__':
    # Get script specific parameters.
    parser = setup_argparse()
    args = parser.parse_args()
    table_name = args.table_name
    database_column_name = args.data_column_name
    data_output_dir = args.output_dir_name if args.output_dir_name else table_name
    embedding_target = args.emb_target
    text_filtering_option = args.filtering_option
    debug = args.enable_debug

    # Get general parameters.
    with open('./parameters.json') as params_fp:
        params = json.load(params_fp)

    batch_size = params['embedding_parameters']['ngram_batch_size'] if embedding_target == 'ngram' else params['embedding_parameters']['document_batch_size']
    num_procs = params['script_parameters']['num_processes']

    db_path = params['input_data']['database_path']
    emb_root_path = params['input_data']['embedding_root_path']
    emb_type = PreTrainedEmbeddings.map_string(params['input_data']['embedding_type'])
    emb_dim = params['script_parameters']['embedding_dimension']
    emb_output_dir = params['generated_data']['embedding_output_root_dir']

    filter_arg = 'pos-filter' if text_filtering_option else None

    # Read data and set up parameters for embedding operations.
    tokenizer = simple_tokenizer # used to consistently tokenize documents

    db_handler = DatabaseHandler(db_path)
    max_doc_len = db_handler.get_longest_document_length(
        table_name, 
        database_column_name,
        tokenizer,
        batch_size)
    
    batched_data = db_handler.read(
        table_name,
        chunksize=batch_size,
        retry=True
    )
    print(f'Read data from database using {batch_size} element batches from {table_name}.')
    if debug: print(f'[DEBUG] Type of batched_data: {type(batched_data)}')

    # Embed the data.
    output_emb_path = f'{emb_output_dir}/{data_output_dir}/emb_{table_name}/'
    output_emb_path = output_emb_path.replace('//', '/').replace('=', '_')

    generate_matrix_partial = partial(
        generate_ngram_matrix,
        max_doc_len=max_doc_len,
        tokenizer=tokenizer,
        embedding_rootpath=emb_root_path,
        embedding_type=emb_type,
        embedding_dimension=emb_dim
    )

    save_embeddings_partial = partial(
        save_embeddings,
        directory_path=output_emb_path,
        dry_run=debug
    )
    
    print('Embedding input documents...')
    for i, batch in enumerate(batched_data):
        adjusted_idx = batch.index + (i * batch_size)

        # Split the current batch into chunks for multiprocessing.
        batched_doc = split_chunks(batch.loc[:, database_column_name], num_procs)
        batched_idx = split_chunks(adjusted_idx, num_procs)

        with Pool(num_procs) as p:
            # For each chunk, embed the data and save to a .npy file.
            grouped_embeddings = p.map(generate_matrix_partial, batched_doc)
            p.starmap(
                save_embeddings_partial,
                zip(grouped_embeddings, batched_idx)
            )
        
        print(f'Batch {i} complete.\n\tBatch starting index: {adjusted_idx[0]}\n\tBatch ending index: {adjusted_idx[-1]}\n')
    
    if debug: print(f'[DEBUG] Total number of rows in the database table: {db_handler.get_table_length(table_name)}')

    print(f'Embedding completed. (Total number of batches: {i}).')
    print(f'Saved embedded text at output path: {output_emb_path}')

