import joblib
import numpy as np
from django.apps import AppConfig
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
CURRENT_DIR = os.path.dirname(__file__)


class MODEL:
    '''
    implements tf 3 layer model 
    {dense, relu, dense,  relu, dense, sigmoid} X {80000, 256, 64, 10}
    '''

    def __init__(self):
        self.first_layer_weights = joblib.load(
            CURRENT_DIR+'/MODEL/first_layer_weights')
        self.first_layer_biases = joblib.load(
            CURRENT_DIR+'/MODEL/first_layer_biases')

        self.second_layer_weights = joblib.load(
            CURRENT_DIR+'/MODEL/second_layer_weights')
        self.second_layer_biases = joblib.load(
            CURRENT_DIR+'/MODEL/second_layer_biases')

        self.third_layer_weights = joblib.load(
            CURRENT_DIR+'/MODEL/third_layer_weights')
        self.third_layer_biases = joblib.load(
            CURRENT_DIR+'/MODEL/third_layer_biases')

        self.tokenizer1 = joblib.load(CURRENT_DIR+'/MODEL/tokenizer1.vect')
        self.tokenizer2 = joblib.load(CURRENT_DIR+'/MODEL/tokenizer2.vect')

    def predict(self, vals):
        """
        takes list of sentence
        """

        vals = [' '.join(re.findall("[a-zA-Z\s\']+", vals[0].lower()))]

        TEST = np.hstack((self.tokenizer1.transform(vals).todense(),
                          self.tokenizer2.transform(vals).todense()))

        # dense 1
        w = np.empty([len(TEST), self.first_layer_weights.shape[1]])
        TEST.dot(self.first_layer_weights, out=w)

        w += self.first_layer_biases
        TEST = np.maximum(w, 0)

        # dense 2
        w = np.empty([len(TEST), self.second_layer_weights.shape[1]])
        TEST.dot(self.second_layer_weights, out=w)

        w += self.second_layer_biases
        TEST = np.maximum(w, 0)

        # dense 3
        w = np.empty([len(TEST), self.third_layer_weights.shape[1]])
        TEST.dot(self.third_layer_weights, out=w)

        def sigmoid(X):
            return 1 / (1 + np.exp(-X))

        w += self.third_layer_biases
        out = sigmoid(w)
        return ("positive" if np.argmax(out) > 5 else "negative",
                np.argmax(out) + 1)


class RaterConfig(AppConfig):
    name = 'rater'
    model = MODEL()
