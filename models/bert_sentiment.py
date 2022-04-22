'''
This file contains code to run a pre-trained BERT sentiment model.

Reference:
https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
'''

from models.base_models import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentBERT(BaseModel):
    def __init__(self):
        self._model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self._tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    def predict(self, texts, get_argmax=True):
        '''
        `texts` is expected to either be a single string or an iterable of strings.
        '''
        tokenized_input = self._tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        del texts
        raw_output = self._model(**tokenized_input)[0].detach().numpy()
        if not get_argmax: return raw_output
        else: return np.argmax(raw_output, axis=1)