'''
This file contains a series of Convolutional Neural Networks (CNNs).

Most of these models are based on models introduced in the below papers.

References:
- Collobert et al. 'Natural Language Processing (Almost) from Scratch" (2011)
- Kim "Convolutional Neural Networks for Sentence Classiﬁcation" (2014)
https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dropout, Dense, Concatenate, GlobalMaxPool1D

from models.base_models import DeepNeuralNetworkEmbeddingTargetPredictor, DeepNeuralNetworkTargetPredictor

class KimCNN(DeepNeuralNetworkTargetPredictor):
    '''
    This model is based on the CNN used in Kim 2015.
    '''
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        c1 = Conv1D(128, 5, padding='same')(in_layer)
        conv_final = Conv1D(64, 3, padding='same')(c1)
        gp = GlobalMaxPool1D()(conv_final)
        dropout = Dropout(0.5)(gp)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model = Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer,
            loss=self._model_loss,
            loss_weights = self._loss_weights)
        return model

class KimCNNEmb(DeepNeuralNetworkEmbeddingTargetPredictor):
    '''
    This model is based on the CNN used in Kim 2015.
    '''
    def _compile_model(self):
        in_layer = Input(shape=(self._n,), dtype='int32')
        embed_layer = self._embedding_layer(in_layer)
        c1 = Conv1D(128, 5, padding='same')(embed_layer)
        conv_final = Conv1D(64, 3, padding='same')(c1)
        gp = GlobalMaxPool1D()(conv_final)
        dropout = Dropout(0.5)(gp)
        out_layer = Dense(self._output_size, activation='softmax')(dropout)

        model = Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer,
            loss=self._model_loss,
            loss_weights = self._loss_weights)
        return model

class SingleInputTargetPredictor(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        c1 = Conv1D(64, 3, padding='same')(in_layer)
        c2 = Conv1D(32, 3, padding='same')(c1)
        c3 = Conv1D(16, 3, padding='same')(c2)
        fl = Flatten()(c3)
        dr1 = Dropout(0.2)(fl)
        d1 = Dense(180, activation='sigmoid')(dr1)
        dr2 = Dropout(0.2)(d1)
        out_layer = Dense(self._output_size, activation='softmax')(dr2)

        model = Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss, 
            loss_weights=self._loss_weights)
        return model

class DualInputTargetPredictor(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        main_in = Input(shape=(self._n, self._emb_dim), name='main_in')
        c1 = Conv1D(64, 3, padding='same')(main_in)
        c2 = Conv1D(32, 3, padding='same')(c1)
        c3 = Conv1D(16, 3, padding='same')(c2)
        fl = Flatten()(c3)
        dr1 = Dropout(0.2)(fl)
        pos_in = Input(shape=(1,), name='pos_in')
        cat = Concatenate(axis=-1, name='cat')([dr1, pos_in])
        d = Dense(64, activation='relu')(cat)
        dr2 = Dropout(0.2)(d)
        main_out = Dense(self._output_size, activation='softmax')(dr2)

        model = Model([main_in, pos_in], main_out)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class DualInputLossTargetPredictor(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        main_in = Input(shape=(self._n, self._emb_dim), name='main_in')
        c1 = Conv1D(64, 3, padding='same')(main_in)
        c2 = Conv1D(32, 3, padding='same')(c1)
        c3 = Conv1D(16, 3, padding='same')(c2)
        fl = Flatten()(c3)
        dr1 = Dropout(0.2)(fl)
        aux_out = Dense(self._output_size, activation='softmax', name='aux_out')(dr1)

        pre_dense = Dense(64, activation='relu', name='pre_input_dense')(dr1)
        pos_in = Input(shape=(1,), name='pos_in')
        cat = Concatenate(axis=-1)([pre_dense, pos_in])
        d = Dense(64, activation='relu')(cat)
        dr2 = Dropout(0.2)(d)
        main_out = Dense(self._output_size, activation='softmax', name='main_out')(dr2)

        model = Model([main_in, pos_in], [main_out, aux_out])
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model