'''
This file defines a series of recurrent neural networks.

Many of these models are based on previously published literature, especially:
- Tai et al. "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks" (2015)
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Bidirectional, GRU
from tensorflow.python.keras.layers.core import Dropout

from models.base_models import DeepNeuralNetworkEmbeddingTargetPredictor, DeepNeuralNetworkTargetPredictor

class TaiLSTM(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = LSTM(400)(in_layer)
        # lstm_layer2 = LSTM(400)(lstm_layer)
        dropout = Dropout(0.5)(lstm_layer)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class TaiBidirectionalLSTM(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = Bidirectional(LSTM(168))(in_layer)
        dropout = Dropout(0.5)(lstm_layer)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class TaiLSTMEmb(DeepNeuralNetworkEmbeddingTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n,), dtype='int32')
        embed_layer = self._embedding_layer(in_layer)
        lstm_layer = LSTM(400)(embed_layer)
        # lstm_layer2 = LSTM(400)(lstm_layer)
        dropout = Dropout(0.5)(lstm_layer)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class TaiBidirectionalLSTMEmb(DeepNeuralNetworkEmbeddingTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n,), dtype='int32')
        embed_layer = self._embedding_layer(in_layer)
        lstm_layer = Bidirectional(LSTM(168))(embed_layer)
        dropout = Dropout(0.5)(lstm_layer)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class GRUModel(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        gru_layer = GRU(400)(in_layer)
        dropout = Dropout(0.5)(gru_layer)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class TaiBidirectionalGRU(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = Bidirectional(GRU(168))(in_layer)
        dropout = Dropout(0.5)(lstm_layer)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class TaiBidirectionalGRUEmb(DeepNeuralNetworkEmbeddingTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n,), dtype='int32')
        embed_layer = self._embedding_layer(in_layer)
        lstm_layer = Bidirectional(GRU(168))(embed_layer)
        dropout = Dropout(0.5)(lstm_layer)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class SingleInputTargetLSTM(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = LSTM(1028)(in_layer)
        d1 = Dense(512, activation='relu')(lstm_layer)
        d2 = Dense(128, activation='relu')(d1)
        out_layer = Dense(self._output_size, activation='softmax')(d2)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class DualInputOutputTargetLSTM(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = LSTM(1028)(in_layer)
        d1 = Dense(512, activation='relu')(lstm_layer)
        d2 = Dense(128, activation='relu')(d1)
        aux_out = Dense(self._output_size, activation='softmax', name='aux_out')(d2)

        aux_in = Input(shape=(1,), name='pos_in')
        cat = Concatenate(axis=-1)([d2, aux_in])
        d3 = Dense(1024, activation='relu')(cat)
        main_out = Dense(self._output_size, activation='softmax', name='main_out')(d3)

        model = Model([in_layer, aux_in], [main_out, aux_out])
        model.compile(
            optimizer=self._optimizer,
            loss=self._model_loss,
            loss_weights=self._loss_weights
        )
        return model

class SimpleSingleInputTargetLSTM(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(in_layer)
        out_layer = Dense(self._output_size, activation='softmax')(lstm_layer)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model