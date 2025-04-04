import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

num_folds = 0
seed = 0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class MultiOutputSGDClassifier(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(MultiOutputSGDClassifier, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        # Initialize SGD classifier with multi-output wrapper
        base_sgd = SGDClassifier(loss='log_loss',  
                               penalty='l2',
                               alpha=0.0001,
                               max_iter=1000,
                               tol=1e-3,
                               random_state=seed,
                               class_weight='balanced')
        self.mdl = MultiOutputClassifier(base_sgd)
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        print("The first 5 predictions are:", self.predictions[:5])
        accuracy = []
        
        for true, pred in zip(data.y_test, self.predictions):
            correct = 0
            for t, p in zip(true, pred):
                if t == p:
                    correct += 1
            accuracy.append(correct / len(true))
        
        print("\nClassification Results:")
        print(f"Average accuracy: {np.mean(accuracy)*100:.2f}%")
    def get_accuracy(self,data):
        accuracy = []
        for true, pred in zip(data.y_test, self.predictions):
            correct = sum(t == p for t, p in zip(true, pred))
            accuracy.append(correct / len(true))
        return np.mean(accuracy)*100
    def data_transform(self) -> None:
        pass