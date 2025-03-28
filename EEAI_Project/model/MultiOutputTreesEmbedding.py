from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import numpy as np
import random
from model.base import BaseModel


num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class MultiOutputTreesEmbedding(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(MultiOutputTreesEmbedding, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        
        # Pipeline with embedding + multi-output classifier
        self.mdl = make_pipeline(
            RandomTreesEmbedding(n_estimators=100, random_state=seed),
            MultiOutputClassifier(
                LogisticRegression(class_weight='balanced', random_state=seed)
            )
        )
        self.predictions = None

    def train(self, data) -> None:
        self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        print("The first 5 predictions are:", self.predictions[:5])
        accuracy = []
        for true, pred in zip(data.y_test, self.predictions):
            correct = sum(t == p for t, p in zip(true, pred))
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
        ...