'''
This file defines a series of base abstract classes.
None of the models defined here should be used directly.

All models implemented in this repository should inherit
from one of the models defined here.
'''
from sklearn.svm import SVC
from data_handling.data_generators import BaseNLPGenerator
from tensorflow.keras.layers import Embedding
import analysis.classifier_metrics as cm


class BaseModel():
    '''
    This class defines a basic set of functions.
    '''
    def __init__():
        raise NotImplementedError('This is an abstract base class. Please use a concrete subclass.')

    def train(self, train_data, test_data):
        raise NotImplementedError()
    
    def test(self, test_data, test_labels):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

class BaseDeepNeuralNetwork(BaseModel):
    def __init__(self):
        raise NotImplementedError('This is an abstract base class. Please use a concrete subclass.')

    def _compile_model(self):
        ''' 
        Compiles a Keras model based on method implementation. 
        '''
        raise NotImplementedError

class SKLearnSentimentPredictor(BaseModel):
    '''
    This class is a wrapper for a Support Vector Classifier
    from scikit-learn.
    '''
    def __init__(self, C, kernel='rbf', d=3):
        '''
        Parameters:
        - `C`: regularization parameter for the SVC
        - `kernel`: the type of kernel to use in the SVC
        - `d`: if using a polynomial kernel, the degree of the polynomial
        '''
        self._model = SVC(C=C, kernel=kernel, degree=d)

    def train(self, train_data, train_labels):
        self._model.fit(train_data, train_labels)

    def test(self, test_data, test_labels):
        preds = self._model.predict(test_data)
        prec = cm.calculate_class_precision(1, preds, test_labels)
        rec = cm.calculate_class_recall(1, preds, test_labels)
        fscore = cm.calculate_class_fscore(prec, rec)
        return prec, rec, fscore

    def predict(self, data):
        return self._model.predict(data)

class DeepNeuralNetworkTargetPredictor(BaseDeepNeuralNetwork):
    '''
    Abstract base class for a Keras aspect-detection model. Expects pre-embedded
    data since there is no `Embedding` layer for these models.

    Concrete subclasses should provide an implementation for `self._compile_model()`.
    '''
    def __init__(self, cws, emb_dim, output_size, optimizer='adam', loss='categorical_crossentropy', loss_weights=None):
        self._n = cws
        self._cws = cws
        self._emb_dim = emb_dim
        self._optimizer = optimizer
        self._model_loss = loss
        self._loss_weights = loss_weights
        self._output_size = output_size
        self._model = self._compile_model()

    def train(self, train_data, train_labels, val_data, val_labels, e=5, callbacks=None, batch_generator=None):
        ''' 
        Trains a compiled Keras model.

        Parameters:
        - `train_data`: data used to train the model
        - `train_labels`: actual labels correlated to `train_data` by index
        - `val_data`: validation data used to track model performance during training
        - `val_labels`: actual labels for `val_data`, also correlated by index
        - `e`: number of epochs to train the Keras model
        - `callbacks`: iterable of Keras callbacks to be used during training; specific training metrics can be defined here
        - `batch_generator`: a batch generating function that must provide batched training data, training
        labels and can provide validation data and labels; if provided, the first four parameters are ignored
        ''' 
        if batch_generator == None:
            return self._model.fit(
                train_data, train_labels,
                validation_data=(val_data, val_labels),
                epochs=e,
                callbacks=callbacks)
        else:
            return self._model.fit_generator(batch_generator, epochs=e, callbacks=callbacks)

    def test(self, test_data, test_labels, callbacks=None):
        '''
        Tests a pre-trained Keras model using the provided data and labels.

        Parameters:
        - `test_data`: data used to test the model
        - `test_labels`: actual labels correlated to `test_data` by index
        - `callbacks`: iterable of Keras callbacks to allow for custom metric definitions
        '''
        return self._model.evaluate(test_data, test_labels, callbacks=callbacks)

    def predict(self, data, batch_size=None):
        ''' 
        Runs a pre-trained Keras model on the provided data and returns
        predicted labels.

        Parameters:
        - `data`: data to be run through the model
        - `batch_size`: number of elements to be run through the model at once
        '''
        return self._model.predict(data, batch_size=batch_size)

class DeepNeuralNetworkEmbeddingTargetPredictor(BaseDeepNeuralNetwork):
    '''
    Abstract base class for a Keras aspect-detection model that uses an `Embedding`
    layer.

    Concrete subclasses should provide an implementation for `self._compile_model()`.
    '''
    def __init__(self, max_seq_length, emb_dim, is_embedding_trainable, num_unique_words, embedding_matrix, output_size):
        self._n = max_seq_length
        self._emb_dim = emb_dim
        self._output_size = output_size
        self._optimizer = 'adam'
        self._model_loss = 'categorical_crossentropy'
        self._loss_weights = None
        self._embedding_layer = self._create_embedding_layer(num_unique_words, embedding_matrix, is_embedding_trainable)
        self._model = self._compile_model()
    
    def _create_embedding_layer(self, num_unique_words, embedding_matrix, trainable_embedding=False):
        '''
        Creates an `Embedding` layer given the provided model hyperparameters.

        This embedding layer should be used in `self._compile_model()` for the model's
        input(s).

        Parameters:
        - `num_unique_words`: the number of unique words in the training corpus
        - `embedding_matrix`: the full set of word embeddings
        - `trainable_embedding`: boolean that determines if the provided word embeddings
        can be updated during training
        '''
        return Embedding(
            num_unique_words + 1, 
            self._emb_dim, 
            weights = [embedding_matrix],
            trainable = trainable_embedding)

    def train(self, batch_generator, epochs=5):
        ''' 
        Trains a compiled Keras model.

        Parameters:
        - `batch_generator`: a batch generating function that must provide batched training data, training
        labels and can provide validation data and labels
        - `epochs`: number of epochs to train the Keras model
        - `callbacks`: iterable of Keras callbacks to be used during training; specific training metrics can be defined here
        ''' 
        self._model.fit_generator(batch_generator, epochs=epochs, callbacks=None)

    def test(self, test_data, test_labels, callbacks=None):
        '''
        Tests a pre-trained Keras model using the provided data and labels.

        Parameters:
        - `test_data`: data used to test the model
        - `test_labels`: actual labels correlated to `test_data` by index
        - `callbacks`: iterable of Keras callbacks to allow for custom metric definitions
        '''
        return self._model.evaluate(test_data, test_labels, callbacks=callbacks)

    def predict(self, data, batch_size=None):
        ''' 
        Runs a pre-trained Keras model on the provided data and returns
        predicted labels.

        Parameters:
        - `data`: data to be run through the model
        - `batch_size`: number of elements to be run through the model at once
        '''
        return self._model.predict(data, batch_size=batch_size)


class UnsupervisedDeepNeuralNetworkModel(BaseDeepNeuralNetwork):
    '''
    Abstract base class for an unsupervised deep neural network.

    Concrete subclasses should provide an implementation for `self._compile_model()`.
    '''
    def __init__(self, win_size, emb_dim, output_size, optimizer='adam', loss='categorical_crossentropy', loss_weights=None):
        self._n = win_size
        self._emb_dim = emb_dim
        self._optimizer = optimizer
        self._model_loss = loss
        self._loss_weights = loss_weights
        self._output_size = output_size

        self._model = self._compile_model()

    def train(self, in_data, e, callbacks=None, batch_generator: BaseNLPGenerator = None):
        ''' 
        Trains a compiled Keras model.

        Parameters:
        - `in_data`: data to "train" the unsupervised model on 
        - `e`: number of epochs to train the Keras model
        - `callbacks`: iterable of Keras callbacks to be used during training; specific training metrics can be defined here
        - `batch_generator`: a batch generating function that must provide batched training data, training
        labels and can provide validation data and labels; if provided, the first parameter is ignored
        ''' 
        if batch_generator is not None:
            self._model.fit_generator(batch_generator, epochs=e, callbacks=callbacks)
        else:
            self._model.fit(in_data, epochs=e, callbacks=callbacks)

    def test(self, test_data, callbacks=None):
        '''
        Tests a pre-trained Keras model using the provided data.
        '''
        raise NotImplementedError()

    def predict(self, data):
        '''
        Runs a pre-trained Keras model on the provided data and returns
        predicted labels.
        '''
        raise NotImplementedError()