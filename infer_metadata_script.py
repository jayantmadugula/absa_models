''' This script uses pre-trained models to output metadata inferences. '''

import argparse
from datetime import datetime
from enum import Enum
import json
import os

import numpy as np
from models.base_models import BaseModel
from utilities.script_helpers import SupportedDatasets
from models.bert_sentiment import SentimentBERT
from database_utilities.database_handler import DatabaseHandler
from tensorflow.keras.models import load_model


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

    # Get output parameters.
    output_root_dir = params_dict['generated_data']['metadata_predictions_root_dir']
    output_filename = f'{args.model}_{args.dataset_name}/{datetime.now()}'

    try:
        os.mkdir(output_root_dir + f'{args.model}_{args.dataset_name}')
    except:
        pass # if the directory already exists, continue

    # Get model parameters.
    batch_size = params_dict['inference_parameters']['batch_size']

    # Set data parameters.
    db_path = params_dict['input_data']['database_path']

    n = int((args.ngram_size - 1) / 2)

    dataset_type = SupportedDatasets(args.dataset_name)
    table_name = f'{dataset_type.value}_n={n}'

    # Get data for inference.
    db_handler = DatabaseHandler(db_path)
    data_iter = (b['ngram'].to_list() for b in db_handler.read(table_name, chunksize=batch_size, columns=[args.data_column_name]))

    # Load model and run inference.
    model = None
    try:
        match SupportedModels(args.model):
            case SupportedModels.BERT_SENTIMENT:
                model = SentimentBERT()
                print('Successfully loaded SentimentBERT model.')
    except ValueError:
        # Load model from path.
        print('Model argument not a member of SupportedModels. Assuming the provided argument is the filepath to a pre-trained Keras model.')
        tf_model = load_model(args.model)
        model = BaseModel(tf_model)
        print('Successfully loaded model as a BaseModel instance.')

    all_preds = []
    for i, batch in enumerate(data_iter):
        batch_preds = model.predict(batch)
        print(f'Predictions complete ({len(batch)}): {i}')
        all_preds.append(batch_preds)

        if isinstance(model, SentimentBERT) and i % 20 == 0:
            # TODO: hack to fix memory issue
            # no impact on predictions since the same pre-trained weights are used
            del model
            model = SentimentBERT()

    np.save(output_root_dir + output_filename, np.concatenate(all_preds))
    print(f'Script complete! Predictions were written to: {output_root_dir + output_filename}')