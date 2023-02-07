'''
This file contains functions used to load embedding dictionaries,
embed text onto an n-dimensional space using loaded dictionaries,
and help validate an embedding.
'''

from enum import Enum
from typing import Callable, Dict, Iterable
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization


class PreTrainedEmbeddings(Enum):
    '''
    Defines supported pre-trained embeddings.
    The value of each enum variant should provide
    a way to access the pre-trained embeddings
    themselves.
    '''
    GLOVE = 'glove.6B.{}d.txt'

    @classmethod
    def map_string(cls, s: str):
        if s.lower() == 'glove':
            return PreTrainedEmbeddings.GLOVE
        else:
            raise ValueError(f'There is no implementation for: {s}.')

    def validate_embedding_dimension(self, embedding_dimension: int) -> bool: 
        if (self is PreTrainedEmbeddings.GLOVE):
            return embedding_dimension in {50, 100, 200, 300}
        else:
            raise ValueError(f'There is no implementation for: {self}.')

def get_embedding_dictionary(
    embedding_rootpath: str,
    embedding_type: PreTrainedEmbeddings,
    embedding_dimension: int):
    '''
    Builds a pre-trained embedding dictionary. The file
    at `embedding_path` should contain the pre-trained embeddings
    for a specific dimension.
    '''
    emb_filepath = _build_pretrained_embedding_filepath(
        embedding_rootpath, 
        embedding_type, 
        embedding_dimension=embedding_dimension)

    emb_file = open(emb_filepath)
    emb_dict = _build_pretrained_embedding(emb_file)
    emb_file.close()
    return emb_dict

def get_embedding_index(
    embedding_rootpath: str,
    embedding_type: PreTrainedEmbeddings,
    embedding_dimension: int):
    '''
    Generates an embedding index for Keras Embedding layers.
    '''
    emb_filepath = _build_pretrained_embedding_filepath(
        embedding_rootpath, 
        embedding_type, 
        embedding_dimension=embedding_dimension)
    emb_file = open(emb_filepath)

    embeddings_index = {}
    for i, line in enumerate(emb_file):
        values = line.split()
        word = values[0]
        embeddings_index[word] = i
    emb_file.close()
    return embeddings_index

def embed_phrases(
    phrases: Iterable[str],
    embedding_rootpath: str,
    embedding_type: PreTrainedEmbeddings,
    embedding_dimension: int) -> pd.DataFrame:
    '''
    Generates embeddings for each entry of `phrases`.

    The returned `DataFrame` includes both tokenized phrases 
    (column name: `phrases`) and the embedded versions of each 
    phrase (column name: `embedded`).
    '''
    emb_filepath = _build_pretrained_embedding_filepath(
        embedding_rootpath, 
        embedding_type, 
        embedding_dimension=embedding_dimension)
    f = open(emb_filepath)
    emb_dict = _build_pretrained_embedding(f)

    embedded_phrases = []
    processed_phrases = []
    for phrase in phrases:
        if len(phrase.split(' ')) > 1:
            # Embed each token in `phrase`.
            subphrases = phrase.split(' ')
            embedded_phrases.extend([_embed_phrase(p, emb_dict, embedding_dimension) for p in subphrases])
            processed_phrases.extend(subphrases)
        else:
            # Only one token in the phrase, so embed only once.
            embedded_phrases.append(_embed_phrase(phrase, emb_dict, embedding_dimension))
            processed_phrases.append(phrase)

    f.close()
    return pd.DataFrame.from_dict({'phrases': processed_phrases, 'embedded': embedded_phrases})

def generate_ngram_matrix(
    texts: Iterable[str],
    max_doc_len: int,
    tokenizer: Callable[[str], int],
    embedding_rootpath: str,
    embedding_type: PreTrainedEmbeddings,
    embedding_dimension: int,
    pad_word='inv') -> np.ndarray:
    '''
    Builds a 2D matrix representation of the inputed `texts`
    using pretrained GloVe word embeddings.

    Returns a numpy matrix with shape `(n, emb_dim)`
    where `n == len(texts)`
    '''
    emb_filepath = _build_pretrained_embedding_filepath(
        embedding_rootpath, 
        embedding_type, 
        embedding_dimension=embedding_dimension)
    f = open(emb_filepath)
    emb_dict = _build_pretrained_embedding(f)

    line_vecs = []
    for line in texts:
        vecs = []
        words = tokenizer(line)
        for word in words:
            vec = _embed_phrase(
                word,
                emb_dict,
                embedding_dimension,
                pad_word
            )
            vecs.append(vec)

        doc_len = len(vecs)
        if (max_doc_len - doc_len) > 0:
            zero_arrs = [np.zeros(embedding_dimension)] * (max_doc_len - doc_len)
            vecs.extend(zero_arrs)

        line_vec = np.stack(vecs)
        line_vecs.append(line_vec)
    
    return np.stack(line_vecs)

def generate_vocabulary_matrix(
        trained_vectorizer: TextVectorization,
        num_tokens: int,
        embedding_type: PreTrainedEmbeddings,
        embedding_rootpath: str,
        embedding_dimension: int
    ):
    '''
    Given a `word_index`, which should come from `TextVectorizer.word_index`, returns a
    dictionary mapping each word to it's embedded representation.

    Docs: https://keras.io/examples/nlp/pretrained_word_embeddings/
    '''
    word_index = dict(zip(trained_vectorizer.get_vocabulary(), range(num_tokens-2)))

    emb_filepath = _build_pretrained_embedding_filepath(
        embedding_rootpath, 
        embedding_type, 
        embedding_dimension=embedding_dimension)
    f = open(emb_filepath)
    emb_dict = _build_pretrained_embedding(f)

    embedding_matrix = np.zeros((num_tokens, embedding_dimension))
    for word, i in word_index.items():
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def flatten_sentence_vectors(word_matrix: np.ndarray) -> np.ndarray:
    '''
    Creates a flattened 1D vector per sentence, with 
    length equal to number of words in the sentence and
    the embedding dimension.
    
    Requires a word matrix as input.
    
    in_shape: (?, x, y)
    out_shape: (?, x * y)
    '''
    new_vecs = []
    for sent_vec in word_matrix:
        r, c = sent_vec.shape
        new_vec = sent_vec.reshape((r*c))
        new_vecs.append(new_vec)
    
    return np.stack(new_vecs)

def check_embedding(text: str, text_embedding: np.ndarray, emb_dict: Dict[str, np.ndarray]) -> bool:
    '''
    Simple check to make sure an inputed sentence
    or text fragment correctly matches a matrix
    of word vectors.
    
    Parameters:
    - `text`: text fragment
    - `text_embedding`: appended word vectors to check
    - `emb_dict`: dictionary containing embeddings
    '''
    for word, word_emb in zip(text.split(), text_embedding):
        if (emb_dict[word] != word_emb).all(): return False
    return True

def create_keras_vectorizer(batched_texts: Iterable[Iterable[str]], max_document_length: int):
    '''
    Returns a `TextVectorizer` object with a vocabulary covering all unique words in `texts`.

    Currently, this function assumes `texts` contains batched data.
    
    `max_document_length` is the maximum number of words in a single document contained in `texts`.
    '''
    print('\nCreating a Keras TextVectorization object...')
    unique_words = set()
    for batch in batched_texts:
        for e in batch:
            [unique_words.add(w) for w in e.split(' ')]

    return TextVectorization(output_sequence_length=max_document_length, vocabulary=list(unique_words))

def _embed_phrase(
    phrase: str, 
    emb_dict: Dict[str, np.ndarray], 
    emb_dim: int, 
    pad_word: str = None) -> np.ndarray:
    '''
    Provides the embedding for a given phrase.

    If the given phrase cannot be found in the pre-trained embeddings,
    a zero-array of dimension `emb_dim` is returned.
    '''
    if pad_word is not None and phrase == pad_word:
        return np.zeros(emb_dim)
    elif phrase in emb_dict:
        return emb_dict[phrase]  
    else:
        return np.zeros(emb_dim)

def _build_pretrained_embedding(f) -> Dict[str, np.ndarray]:
    '''
    Builds pretrained embedding dictionary from the inputed GloVe file `f`.

    The returned dictionary has the following schema:
    - key: word
    - value: associated word vector
    '''
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    return embeddings_index

def _build_pretrained_embedding_filepath(
    rootpath: str, 
    embedding_type: PreTrainedEmbeddings,
    **kwargs):
    '''
    Builds the path to a pre-trained embedding file given a rootpath,
    the type of embedding being used, and any specific information required
    by that embedding type.

    For `GloVe` embeddings, `embedding_dimension` must be provided in `kwargs`.
    '''
    if embedding_type == PreTrainedEmbeddings.GLOVE:
        embedding_dimension = kwargs['embedding_dimension']
        
        # Make sure the provided embedding dimensions is valid.
        if not embedding_type.validate_embedding_dimension(embedding_dimension):
            raise ValueError(f'Invalid embedding dimension, {embedding_dimension}, provided.')
        
        return rootpath + embedding_type.value.format(embedding_dimension)
    else:
        raise NotImplementedError('Only GLOVE pre-trained vectors are supported right now.')