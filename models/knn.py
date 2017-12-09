import math
import os
import sys
sys.path.append("./")

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from models.ocr_model import OCRModel

class OCRKNN(OCRModel):
    def __init__(self, image_dir="images/", debug=False, n_neighbors=5):
        super(OCRKNN, self).__init__(image_dir=image_dir, debug=debug)

        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self):
        training_feats, training_labels = self.training_data()

        print("Fitting classifier...")
        self.classifier.fit(training_feats, training_labels)
        print("Fitted classifier.")

    def eval(self):
        eval_feats, eval_labels = self.eval_data()

        print("Evaluating data points with classifier...")
        predictions = self.classifier.predict_proba(eval_feats)
        print("Evaluated data points with classifier.")

        return predictions

