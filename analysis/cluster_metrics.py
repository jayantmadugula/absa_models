'''
This file contains metrics used to understand clustering model performance.
'''

from typing import Iterable
import numpy as np
from scipy.spatial.distance import cosine

# General cluster metrics.
def calculate_cluster_purity(cluster_labels, true_labels):
    '''
    Computes the total number of "correct" cluster assignments divided
    by the number of clusters.

    Parameters:
    - `cluster_labels`: iterable of predicted cluster assignments
    - `true_labels`: iterable of actual data labels
    '''
    max_labels_per_cluster = []
    clusters = cluster_labels.unique()
    for c in clusters:
        # Find actual labels of the data points in the current cluster.
        data_in_cluster = true_labels[cluster_labels==c]
        # Determine the most common actual label of the data in the current cluster.
        most_common_label = data_in_cluster.mode()[0]
        # Compute the number of elements in the current cluster with the most common label.
        max_labels_per_cluster.append(data_in_cluster[data_in_cluster == most_common_label].shape[0])
    # Compute the total number of "correct" cluster assignments divided by the number of clusters.
    return sum(max_labels_per_cluster) / len(clusters)

# NLP-specific cluster metrics.
def calc_coherence_score(most_common: Iterable[str], texts: Iterable[str]):
    '''
    Calculates the coherence score for an aspect-detection model. The computation is
    based on the formula presented in He et al.'s "An Unsupervised Neural Attention
    Model for Aspect Extraction" and is specific to a single cluster or aspect. \\
    The coherence score is based on the co-occurence frequency of two top words divided
    by the occurence of one of those two top words. The log of this division is then taken
    before being added to a rolling sum over each possible combination of top words.

    Parameters:
    - `most_common`: the top words for a single cluster or aspect should be an iterable of "top words" from a cluster or aspect
    - `texts`: a full corpus
    '''
    score = 0
    for n in range(1, len(most_common)):
        w_n = most_common[n]
        for l in range(0, n):
            w_l = most_common[l]

            # Calculate frequency of `w_l`
            occurence_frequency = _calc_doc_frequency(w_l, texts)
            # Calculate cooccurence of `w_n` and `w_l`
            cooccurence_frequency = _calc_cooccurence_doc_frequency(w_n, w_l, texts)

            # No penalty for words that don't appear
            if occurence_frequency == 0: continue
            score += np.log((cooccurence_frequency + 1) / (occurence_frequency))

    return score

def _calc_doc_frequency(w: str, texts: Iterable[str]):
    '''
    Counts the number of documents in the corpus (`texts`) where word
    `w` occurs. This count is divided by the number of documents in the
    corpus to provide a final frequency.

    Parameters:
    - `w`: a word
    - `texts`: the corpus to search for occurences of `w`
    '''
    freq = 0
    for t in texts:
        if w in t: freq += 1
    return freq / len(texts)

def _calc_cooccurence_doc_frequency(w: str, v: str, texts: Iterable[str]):
    '''
    Counts the number of documents in the corpus (`texts`) where words `w`
    and `v` are both present. The count is divided by the number of documents
    in the corpus to provide the final co-occurence frequency.
    
    Parameters:
    - `w`: a word present in the corpus
    - `v`: a word present in the corpus
    - `texts`: the corpus to search for co-occurences of `w` and `v`
    '''
    freq = 0
    for t in texts:
        if w in t and v in t: freq += 1
    return freq / len(texts)

def find_top_words(model, emb_dict, unique_words, n=10, display=True):
    '''
    Based on He et al.'s "An Unsupervised Neural Attention Model for Aspect Extraction",
    determines the `n` words most associated with each aspect learned by the `model`.
    Only the words in `unique_words` are considered for this calculation. If a word in
    `unique_words` cannot be found in `emb_dict.keys()`, it is ignored.

    The strength of a word's association with an aspect is determined by the cosine
    distance between the word and aspect (in the word embedding space).

    Parameters:
    - `model`: a pre-trained Tensorflow model with an `aspect_emb` layer
    - `emb_dict`: the full set of word embeddings available to the model
    - `unique_words`: the set of unique words to be considered in the computation
    - `n`: the number of top words per aspect to be found
    - `display`: when set to `True`, the top `n` words per aspect are printed to stdout
    '''
    aspect_emb_weights = model._model.get_layer('aspect_emb').get_weights()[0]
    top_words = {}
    for i in range(0, aspect_emb_weights.shape[0]):
        c_top_words = [(cosine(emb_dict[w], aspect_emb_weights[i]), w) for w in unique_words if w in emb_dict]
        c_top_words.sort()
        c_top_words.reverse()
        top_words[i] = c_top_words[:n]

    if display:
        for i in range(0, aspect_emb_weights.shape[0]):
            print('Topic {}: {}\n'.format(i, [w for _, w in top_words[i]]))
        
    return top_words