'''
TEMP: The contents of this script should be moved to classes. Do NOT have a script for every single model...
'''

from datetime import datetime
import json
import os
from typing import Dict
from data_handling import data_loaders, data_generators
from database_utilities.database_handler import DatabaseHandler
from models.abae_models import *
from utilities.script_helpers import SupportedDatasets
from data_handling.document_tokenizers import simple_tokenizer
from data_handling import embedding_generation as eg
import numpy as np
import argparse
from enum import Enum
from sklearn.preprocessing import OneHotEncoder

class SupportedAspectModels(Enum):
    SIMPLE_ABAE = 'SimpleABAE'
    NEW_ABAE = 'New_ABAE'
    ABAE_T = 'ABAE_T'
    ABAE_O = 'ABAE_O'
    ABAE_A = 'ABAE_A'
    ABAE_ALSTM = 'ABAE_ALSTM'


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='''
        Trains an aspect-detection model based on an auto-encoder architecture.

        This script supports a number of different models. For more information, run $ python train_absa_model_script --help
        ''')
    
    parser.add_argument(
        'dataset_name',
        type=str,
        choices=[e.value for e in SupportedDatasets],
        help='Name of the dataset used for training.'
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
        '--use_embedded_data',
        type=bool,
        default=False,
        required=False,
        help='If True, the model will use pre-embedded training data instead of relying on a Keras Embedding layer. Both scenarios use GloVe embeddings.'
    )

    parser.add_argument(
        '--enable_embedding_training',
        type=bool,
        default=False,
        required=False,
        help='If True, the Keras Embedding layer will not be frozen during training. Only works when --use_embedded_data is False.'
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
        '-m',
        '--metadata',
        default=None,
        required=False,
        help='If None, no metadata is used for model training. Otherwise, provide a filepath to a .npy file containing the required metadata.'
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
    print(f'Model trained with settings:\n{model_train_settings}\n')
    with open((settings_path := f'{model_save_path}/train_settings.json'), 'w') as fp:
        json.dump(model_train_settings, fp)
        print(f'Saved model training settings at path: {settings_path}')

def determine_metadata(model_type: SupportedAspectModels, dataset_type: SupportedDatasets, metadata_arg: str):
    '''
    Returns the tuple `(metadata, is_metadata_copied)`.

    Parameters:
    - `metadata` can be `None`, an Iterable, or a filepath.
    - `is_metadata_copied` is a boolean.
    - `metadata_arg` should be the command line argument passed in for the metadata flag
    '''
    match (model_type, dataset_type):
        case (SupportedAspectModels.SIMPLE_ABAE, _) | (SupportedAspectModels.NEW_ABAE, _):
            return (None, True)
        case SupportedAspectModels.ABAE_T, _:
            # Need 1-dimensional metadata for additional input.
            arr = np.load(metadata_arg)

            if len(arr.shape) == 1:
                return (arr, True)
            elif len(arr.shape) == 2:
                print(f'Found array with shape: {arr.shape}. Applying argmax to get single-dimension metadata for ABAE_T.')
                return (np.argmax(arr, axis=1), True)
            else:
                print(f'Read array from {metadata_arg} and found invalid shape {arr.shape}.')
                raise ValueError('Invalid metadata shape for ABAE_T.')
        case SupportedAspectModels.ABAE_O | SupportedAspectModels.ABAE_A | SupportedAspectModels.ABAE_ALSTM, _:
            # Need multi-dimensional metadata for additional input.
            arr = np.load(metadata_arg)

            if len(arr.shape) == 1:
                # Turn into one-hot encoding.
                print('Running OneHotEncoder on single-dimensional metadata...')
                return (OneHotEncoder(handle_unknown='ignore').fit_transform(arr.reshape(-1, 1)).toarray(), True)
            elif len(arr.shape) == 2:
                return (arr, True)
            else:
                print(f'Read array from {metadata_arg} and found invalid shape {arr.shape}.')
                raise ValueError(f'Invalid metadata shape for {model_type}.')
        case _:
            raise ValueError('Invalid model_type and dataset_type combination (for now, at least).')

def create_model(model_type: SupportedAspectModels, use_emb_data: bool, **kwargs):
    valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    print(f'Building and training a {model_type.value} model.')

    match model_type, use_emb_data:
        case SupportedAspectModels.SIMPLE_ABAE, True:
            model = SimpleABAE(**valid_kwargs)
        case SupportedAspectModels.SIMPLE_ABAE, False:
            model = SimpleABAE_Emb(**valid_kwargs)
        case SupportedAspectModels.NEW_ABAE, True:
            model = New_ABAE(**valid_kwargs)
        case SupportedAspectModels.NEW_ABAE, False:
            model = New_ABAE_Emb(**valid_kwargs)
        case SupportedAspectModels.ABAE_T, False:
            model = ABAE_T_Emb(**valid_kwargs)
        case SupportedAspectModels.ABAE_O, False:
            model = ABAE_O_Emb(**valid_kwargs)
        case SupportedAspectModels.ABAE_A, False:
            model = ABAE_A_Emb(**valid_kwargs)
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
    metadata_filepath = args.metadata
    generated_embeddings_dirpath = params_dict['generated_data']['embedding_output_root_dir']
    output_model_dirpath = params_dict['generated_data']['trained_model_output_root_dir']

    # Set script parameters.
    num_procs = params_dict['script_parameters']['num_processes']

    # Set data parameters.
    db_path = params_dict['input_data']['database_path']
    emb_root_path = params_dict['input_data']['embedding_root_path']
    max_tokens = params_dict['training_parameters']['max_tokens']

    # Set model parameters.
    model_type = SupportedAspectModels(args.model)
    use_emb_data = args.use_embedded_data
    train_emb_layer = args.enable_embedding_training if not use_emb_data else None
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
    model_save_path = model_save_path if use_emb_data else model_save_path + '_Emb'

    print(f'Fetching pre-embedded training data from path: {emb_data_dirpath}')
    print(f'Trained model will be saved to path: {model_save_path}')

    num_rows = len([f for f in os.listdir(emb_data_dirpath) if f.endswith('.npy')])
    print(f'\nFound {num_rows} rows in dataset at {emb_data_dirpath}.')

    metadata, metadata_copied = determine_metadata(model_type, dataset_type, metadata_filepath)
    if metadata is None: print('No metadata used for model training.')
    elif metadata.shape[0] != num_rows:
        raise ValueError(f'Inconsistent number of rows in data and metadata inputs ({num_rows} and {metadata.shape[0]}, respectively).') 
    else:
        print(f'Found {metadata.shape[0]} rows of metadata at {metadata_filepath}')

    # Training parameters. These are not always required.
    target_input_size = None
    num_tokens = None
    embedding_matrix = None

    # Set up data loading for model training.
    print('\nSetting up data and model for training.')
    if use_emb_data:
        print(f'Setting up data for {model_type} with pre-saved embedded data at {emb_data_dirpath}')
        if model_type == SupportedAspectModels.SIMPLE_ABAE:
            # Using pre-saved & embedded data.
            data_loader = data_loaders.EmbeddedDataLoader(
                data_path=emb_data_dirpath, 
                embedding_dim=emb_dim, 
                n=n,
                n_procs=num_procs
            )

            data_generator = data_generators.SimpleABAEGenerator(
                data_loader=data_loader, 
                indices=np.arange(num_rows),
                batch_size=batch_size,
                ngram_len=window_len,
                embedding_dim=emb_dim
            )
        else:
            raise ValueError(f'Only {SupportedAspectModels.SIMPLE_ABAE} is supported with pre-embedded input data right now.')
    else:
        print(f'Setting up data for {model_type}.')
        # Get required values for vectorization & embedding.
        db_table_name = f'{dataset_type.value}_n={n}'
        database_column_name = 'ngram' #TODO parameterize

        data_gen = lambda: (batch[database_column_name].tolist() for batch in db_handler.read(
            db_table_name,
            chunksize=batch_size, 
        ))
        
        db_handler = DatabaseHandler(db_path)
        max_doc_len = db_handler.get_longest_document_length(
            db_table_name, 
            database_column_name,
            simple_tokenizer,
            batch_size)

        # Generate a vocabulary for the dataset and build embedding matrix.
        vectorizer = eg.create_keras_vectorizer(data_gen(), max_doc_len)
        num_tokens = len(vectorizer.get_vocabulary()) + 2
        embedding_matrix = eg.generate_vocabulary_matrix(
            vectorizer,
            num_tokens,
            eg.PreTrainedEmbeddings.GLOVE,
            emb_root_path,
            emb_dim
        )

        # Create data loader and generator objects.
        data_loader = data_loaders.SqliteDataLoader(
            database_path=db_path,
            table_name=db_table_name,
            data_column_name=database_column_name,
            vectorizer=vectorizer
        )

        if model_type in (SupportedAspectModels.SIMPLE_ABAE, SupportedAspectModels.NEW_ABAE):
            # These models only accept a "positive" and "negative" data input.
            data_generator = data_generators.SimpleABAEGenerator(
                data_loader=data_loader, 
                indices=np.arange(num_rows),
                batch_size=batch_size,
                ngram_len=window_len,
                embedding_dim=emb_dim
            )
        elif model_type in (SupportedAspectModels.ABAE_O, SupportedAspectModels.ABAE_A, SupportedAspectModels.ABAE_ALSTM):
            # Remaining models require a single additional metadata input.
            if metadata_copied:
                metadata_loader = data_loaders.InMemoryDataLoader(metadata)
                target_input_size = metadata.shape[1] if metadata.shape[1] > 1 else None
            else:
                raise ValueError('Currently, only in memory metadata loading is supported.')

            data_generator = data_generators.ABAEMetadataGenerator(
                data_loader,
                metadata_loader=metadata_loader,
                indices=np.arange(num_rows),
                batch_size=batch_size,
                ngram_len=window_len,
                embedding_dim=emb_dim
            )
        else:
            raise ValueError(f'{model_type} models are not implemented yet.')

    # Build and train the model.
    aspect_model = create_model(
        model_type,
        use_emb_data,
        num_tokens=num_tokens,
        emb_matrix=embedding_matrix,
        neg_size=neg_size, 
        win_size=window_len,
        emb_dim=emb_dim, 
        output_size=num_aspects,
        target_input_size=target_input_size,
        trainable_emb_layer=train_emb_layer
    )
    
    print('\nStarting model training...')
    if not args.debug:
        aspect_model.train(
            in_data=None,
            e=epochs,
            batch_generator=data_generator
        )

        print(f'\n\nModel training complete, saving to {model_save_path}')
        aspect_model._model.save(model_save_path)
        save_model_settings(vars(args), params_dict, model_save_path)
        print(f'Model training completed! The trained model has been saved to: {model_save_path}')

    else:
        print(f'Debug enabled, so no model is trained. Model save path: {model_save_path}')