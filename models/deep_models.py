'''
This files contains "simple" deep neural networks. 
No convolutional, recurrent, or attention layers are used here.

These models are intended to provide a baseline.
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Concatenate

from models.base_models import DeepNeuralNetworkTargetPredictor

class SingleInputTargetNetwork(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        main_in = Input(shape=(self._n, self._emb_dim))
        d1 = Dense(2048, activation='relu')(main_in)
        d2 = Dense(1024, activation='relu')(d1)
        f = Flatten()(d2)
        out = Dense(self._output_size, activation='softmax')(f)

        model = Model(main_in, out)
        model.compile(optimizer=self._optimizer, loss=self._model_loss, loss_weights=self._loss_weights)
        return model

class DualInputOutputTargetNetwork(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        main_in = Input(shape=(self._n, self._emb_dim))
        d1 = Dense(2048, activation='relu')(main_in)
        f = Flatten()(d1)
        aux_out = Dense(self._output_size, activation='softmax', name='aux_out')(f)
        
        aux_in = Input(shape=(1,))
        merge = Concatenate(axis=-1)([f, aux_in])
        d2 = Dense(1024, activation='relu')(merge)
        main_out = Dense(self._output_size, activation='softmax', name='main_out')(d2)
        
        model = Model([main_in, aux_in], [main_out, aux_out])
        model.compile(optimizer=self._optimizer, loss=self._model_loss, loss_weights = self._loss_weights)
        return model