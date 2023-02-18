''' This script uses pre-trained models to output metadata inferences. '''

import argparse
from enum import Enum
import json
from utilities.script_helpers import SupportedDatasets
from models.bert_sentiment import SentimentBERT
from database_utilities.database_handler import DatabaseHandler


class SupportedModels(Enum):
    BERT_SENTIMENT = 'BERT_Sentiment'

def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='''
    Runs pre-trained models that output metadata for further ABSA experimentation.

    Currently, the script supports models that output document-level sentiment tags.
    ''')

    parser.add_argument(
        'model',
        type=str,
        help=f'''
        Name of the pre-trained model to use. 
        
        This can be the path to a saved Keras model that accepts an iterable of texts and outputs a sentiment tag, or one of the following strings, which correspond to pre-trained models: {', '.join([f'"{e.value}"' for e in SupportedModels])}.
    ''')

    parser.add_argument(
        'dataset_name',
        type=str,
        choices=[e.value for e in SupportedDatasets],
        help=f'''
        Name of the dataset used for inference.    
    ''')

    parser.add_argument(
        'data_column_name',
        type=str,
        help='Column name containing the documents to run inference on.'
    )

    parser.add_argument(
        '-n',
        '--ngram_size',
        type=int,
        required=True,
        help='Size of the ngrams the model is training on. For the aspect models defined in this repository, this number must be odd.'
    )

    return parser

def validate_arguments(args):
    ''' Additional validation for input arguments. '''
    if args.ngram_size % 2 == 0:
        raise ValueError('ngram_size must be an odd number')


if __name__ == '__main__':
    print('Starting the infer_metadata_script.')

    # Parse script arguments.
    parser = setup_argparse()
    args = parser.parse_args()
    validate_arguments(args)

    # Read project-wide parameters.
    with open('./parameters.json') as fp:
        params_dict = json.load(fp)

    # Set data parameters.
    db_path = params_dict['input_data']['database_path']

    n = int((args.ngram_size - 1) / 2)

    dataset_type = SupportedDatasets(args.dataset_name)
    table_name = f'{dataset_type.value}_n={n}'

    # Get data for inference.
    db_handler = DatabaseHandler(db_path)
    data_iter = db_handler.read(table_name, chunksize=10, columns=[args.data_column_name])

    # TEMP: hard-coded model & data for testing
    model = SentimentBERT()
    res = model.predict(next(data_iter)['ngram'].to_list())
    print(res)