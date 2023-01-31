'''
TEMP: The contents of this script should be moved to classes. Do NOT have a script for every single model...
'''

from datetime import datetime
import json
import os
from data_handling.embedding_generation import generate_ngram_matrix
from data_handling import data_loaders, data_generators
from models.abae_models import SimpleABAE
import numpy as np
import argparse
from enum import Enum

class SupportedAspectModels(Enum):
    SIMPLE_ABAE = 'SimpleABAE'

class SupportedDatasets(Enum):
    RESTAURANT_REVIEWS = 'restaurant_reviews'

def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='''
        Trains an aspect-detection model based on an auto-encoder architecture.

        This script supports a number of different models. For more information, run $ python train_absa_model_script --help
        ''')
    
    parser.add_argument(
        'dataset_name',
        type=str,
        choices=[e.value for e in SupportedDatasets],
        help='Name of the dataset used for trainng.'
    )

    parser.add_argument(
        'model',
        type=str,
        choices=[e.value for e in SupportedAspectModels],
        help='Type of model to train.'
    )

    parser.add_argument(
        '-n',
        '--ngram_size',
        type=int,
        required=True,
        help='Size of the ngrams the model is training on. For the aspect models defined in this repository, this number must be odd.'
    )

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        required=True,
        help='The number of epochs to train the model. Each epoch trains the full network on every element in the dataset.'
    )

    parser.add_argument(
        '-a',
        '--number_of_aspects',
        type=int,
        default=14,
        required=False,
        help='The number of "classes", or aspects, in the model\'s output. This corresponds to the shape of the model\'s output layer.'
    )


    return parser

def validate_arguments(args):
    ''' Additional validation for input arguments. '''
    if args.ngram_size % 2 != 0:
        raise ValueError('ngram_size must be an odd number')

if __name__ == '__main__':
    print('Starting the train_absa_model_script.')

    # Parse script arguments.
    parser = setup_argparse()
    args = parser.parse_args()

    # Read project-wide parameters.
    with open('./parameters.json') as fp:
        params_dict = json.load(fp)

    # Get data information.
    dataset_name = args.dataset_name
    generated_embeddings_dirpath = params_dict['generated_data']['embedding_output_root_dir']
    output_model_dirpath = params_dict['generated_data']['trained_model_output_root_dir']

    # Set script parameters.
    num_procs = params_dict['script_parameters']['num_processes']

    # Set model parameters.
    model_name = args.model
    emb_dim = params_dict['script_parameters']['embedding_dimension']
    batch_size = params_dict['training_parameters']['batch_size']
    
    window_len = args.ngram_size
    n = int((window_len - 1) / 2)
    epochs = args.epochs
    num_aspects = args.number_of_aspects
    neg_size = batch_size

    print(f'\nParameters for training run:\n\tn: {n}\n\temb_dim: {emb_dim}\n\tbatch_size: {batch_size}\n\tepochs: {epochs}\n')
    print(f'\nParameters for the model:\n\tNumber of aspects: {num_aspects}\n\tNegative input size: {neg_size}\n')

    embs_dirpath = f'{generated_embeddings_dirpath}{dataset_name}_n={n}/emb_{dataset_name}-n_{n}/'
    model_save_path = f'{output_model_dirpath}{dataset_name}_models/{datetime.now()}_{model_name}'

    print(f'Fetching pre-embedded trainng data from path: {embs_dirpath}')
    print(f'Trained model will be saved to path: {model_save_path}')

    num_rows = len([f for f in os.listdir(embs_dirpath) if f.endswith('.npy')])
    print(f'\nFound {num_rows} rows in dataset at {embs_dirpath}.')

    # Set up data loading for model training.
    emb_data_loader = data_loaders.EmbeddedDataLoader(
        data_path=embs_dirpath, 
        embedding_dim=emb_dim, 
        n=n, 
        metadata=None, 
        n_procs=num_procs
    )
    data_generator = data_generators.SimpleABAEGenerator(
        data_loader=emb_data_loader, 
        indices=np.arange(num_rows),
        batch_size=batch_size,
        ngram_len=window_len,
        embedding_dim=emb_dim
    )

    # Build and train the model.
    print('Building and training the NLP model.')
    abae_model = SimpleABAE(neg_size=neg_size, win_size=window_len, emb_dim=emb_dim, output_size=num_aspects)
    abae_model._model.summary()
    print()

    abae_model.train(
        in_data=None,
        e=epochs,
        batch_generator=data_generator
    )

    abae_model._model.save(model_save_path)
    print(f'Model training completed! The trained model has been saved to: {model_save_path}')