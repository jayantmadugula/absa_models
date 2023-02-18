'''
This file contains code to run a pre-trained BERT sentiment model.

Reference:
https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
'''

from models.base_models import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

class SentimentBERT(BaseModel):
    def __init__(self):
        self._model = TFAutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self._tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    def predict(self, texts, get_argmax=True):
        '''
        `texts` is expected to either be a single string or an iterable of strings.
        '''
        tokenized_input = self._tokenizer(texts, return_tensors='tf', padding=True, truncation=True)
        del texts
        raw_output = self._model(tokenized_input)[0].numpy() # output classes: 1 - 5 stars
        if not get_argmax: return raw_output
        else: return np.argmax(raw_output, axis=1) # argmax gives the most likely star rating