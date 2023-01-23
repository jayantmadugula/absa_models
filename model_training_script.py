'''
TEMP: The contents of this script should be moved to classes. Do NOT have a script for every single model...
'''

import os
from database_utilities.database_handler import DatabaseHandler
from data_handling.embedding_generation import generate_ngram_matrix
from data_handling import data_loaders, data_generators
from models.abae_models import SimpleABAE
import numpy as np

if __name__ == '__main__':
    print('Starting the model_training_script')

    # Set constants.
    DB_PATH = '../databases/corpus_database.db'

    # Set script parameters.
    num_procs = 10

    # Set model parameters.
    n = 2
    emb_dim = 200
    batch_size = 25000
    epochs = 1

    num_aspects = 14
    neg_size = batch_size

    print(f'Parameters for training run:\n\tn: {n}\n\temb_dim: {emb_dim}\n\tbatch_size: {batch_size}\n\tepochs: {epochs}\n')
    print(f'Parameters for the model:\n\tNumber of aspects: {num_aspects}\n\tNegative input size: {neg_size}\n')
    print('Computing additional parameters for training...')

    window_len = 2 * n + 1

    embs_dirpath = f'./generated_embeddings/restaurantreviews_n={n}/emb_restaurantreviews_n={n}/'
    db_table_name = f'restaurantreviews_n={n}'
    model_save_path = f'./TRAINED_MODEL'

    # num_rows = DatabaseHandler(DB_PATH).get_table_length(db_table_name)
    num_rows = len([f for f in os.listdir(embs_dirpath) if f.endswith('.npy')])
    print(f'Found {num_rows} rows in the {db_table_name} table.\n')

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