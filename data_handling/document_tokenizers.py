'''
This file contains tokenizers that can be used in scripts and other functions.

Each tokenizer is expected to take a string and output a list of strings, where
each element is a single token.
'''

from typing import Iterable


def simple_tokenizer(doc: str) -> Iterable[str]:
        return doc.split()