''' This script uses pre-trained models to output metadata inferences. '''

import argparse
from enum import Enum


class SupportedModels(Enum):
    BERT_SENTIMENT = 'BERT (Sentiment)'

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

    return parser


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()

    