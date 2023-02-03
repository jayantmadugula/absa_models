'''
TEMP: The contents of this script should be moved to classes. Do NOT have a script for every single model...
'''

from datetime import datetime
import json
import os
from typing import Dict
from data_handling import data_loaders, data_generators
from models.abae_models import New_ABAE, SimpleABAE
import numpy as np
import argparse
from enum import Enum

class SupportedAspectModels(Enum):
    SIMPLE_ABAE = 'SimpleABAE'
    NEW_ABAE = 'New_ABAE'
    ABAE_T = 'ABAE_T'
    ABAE_O = 'ABAE_O'
    ABAE_A = 'ABAE_A'
    ABAE_ALSTM = 'ABAE_ALSTM'

class SupportedDatasets(Enum):
    RESTAURANT_REVIEWS = 'restaurantreviews'

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

    parser.add_argument(
        '-d',
        '--debug',
        type=bool,
        default=False,
        required=False,
        help='If true, no model is trained.'
    )

    return parser

def validate_arguments(args):
    ''' Additional validation for input arguments. '''
    if args.ngram_size % 2 == 0:
        raise ValueError('ngram_size must be an odd number')
    
def save_model_settings(args_dict: Dict, params_dict: Dict, model_save_path: str):
    model_train_settings = {'script_args': args_dict, 'training_parameters': params_dict}
    print(model_train_settings)
    with open((settings_path := f'{model_save_path}/train_settings.json'), 'w') as fp:
        json.dump(model_train_settings, fp)
        print(f'Saved model training settings at path: {settings_path}')

def determine_metadata(model_type: SupportedAspectModels, dataset_type: SupportedDatasets):
    '''
    Returns the tuple `(metadata, is_metadata_copied)`.

    `metadata` can be `None`, an Iterable, or a filepath.
    
    `is_metadata_copied` is a boolean.
    '''
    match (model_type, dataset_type):
        case (SupportedAspectModels.SIMPLE_ABAE, _) | (SupportedAspectModels.NEW_ABAE, _):
            return (None, True)
        case _:
            raise ValueError('Invalid model_type and dataset_type combination (for now, at least).')
        
def create_model(model_type: SupportedAspectModels, **kwargs):
    valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    print(f'Building and training a {model_type.value} model.')

    match model_type:
        case SupportedAspectModels.SIMPLE_ABAE:
            model = SimpleABAE(**valid_kwargs)
        case SupportedAspectModels.NEW_ABAE:
            model = New_ABAE(**valid_kwargs)
        case _:
            raise ValueError(f'Model of type {model_type} is not yet implemented.')
        
    model._model.summary()
    return model

if __name__ == '__main__':
    print('Starting the train_absa_model_script.')

    # Parse script arguments.
    parser = setup_argparse()
    args = parser.parse_args()

    # Read project-wide parameters.
    with open('./parameters.json') as fp:
        params_dict = json.load(fp)

    # Get data information.
    dataset_type = SupportedDatasets(args.dataset_name)
    generated_embeddings_dirpath = params_dict['generated_data']['embedding_output_root_dir']
    output_model_dirpath = params_dict['generated_data']['trained_model_output_root_dir']

    # Set script parameters.
    num_procs = params_dict['script_parameters']['num_processes']

    # Set model parameters.
    model_type = SupportedAspectModels(args.model)
    emb_dim = params_dict['script_parameters']['embedding_dimension']
    batch_size = params_dict['training_parameters']['batch_size']
    
    window_len = args.ngram_size
    n = int((window_len - 1) / 2)
    epochs = args.epochs
    num_aspects = args.number_of_aspects
    neg_size = batch_size

    validate_arguments(args)

    print(f'\nParameters for training run:\n\tn: {n}\n\temb_dim: {emb_dim}\n\tbatch_size: {batch_size}\n\tepochs: {epochs}\n')
    print(f'\nParameters for the model:\n\tNumber of aspects: {num_aspects}\n\tNegative input size: {neg_size}\n')

    # Set computed parameters and data paths for training.
    emb_data_dirpath = f'{generated_embeddings_dirpath}{dataset_type.value}_n_{n}/emb_{dataset_type.value}_n_{n}/'
    model_save_path = f'{output_model_dirpath}{dataset_type.value}_models/{datetime.now()}_{model_type.value}'

    print(f'Fetching pre-embedded training data from path: {emb_data_dirpath}')
    print(f'Trained model will be saved to path: {model_save_path}')

    num_rows = len([f for f in os.listdir(emb_data_dirpath) if f.endswith('.npy')])
    print(f'\nFound {num_rows} rows in dataset at {emb_data_dirpath}.')

    metadata, metadata_copied = determine_metadata(model_type, dataset_type)
    if metadata is not None: print('Additional metadata included in model training.')
    else: print('No metadata used for model training.')

    target_input_size = None

    # Set up data loading for model training.
    emb_data_loader = data_loaders.EmbeddedDataLoader(
        data_path=emb_data_dirpath, 
        embedding_dim=emb_dim, 
        n=n, 
        metadata=metadata, 
        n_procs=num_procs,
        is_metadata_copied=metadata_copied
    )
    data_generator = data_generators.SimpleABAEGenerator(
        data_loader=emb_data_loader, 
        indices=np.arange(num_rows),
        batch_size=batch_size,
        ngram_len=window_len,
        embedding_dim=emb_dim
    )

    # Build and train the model.
    aspect_model = create_model(
        model_type,
        neg_size=neg_size, 
        win_size=window_len,
        emb_dim=emb_dim, 
        output_size=num_aspects,
        target_input_size=target_input_size
    )
    
    if not args.debug:
        aspect_model.train(
            in_data=None,
            e=epochs,
            batch_generator=data_generator
        )

        aspect_model._model.save(model_save_path)
        save_model_settings(vars(args), params_dict, model_save_path)
        print(f'Model training completed! The trained model has been saved to: {model_save_path}')

    else:
        print(f'Debug enabled, so no model is trained. Model save path: {model_save_path}')