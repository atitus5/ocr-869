import math
import os
import sys
sys.path.append("./")

import numpy as np
from sklearn import svm

from models.ocr_model import OCRModel

class OCRSVM(OCRModel):
    def __init__(self, image_dir="images/", debug=False, kernel="rbf", degree=3):
        super(OCRSVM, self).__init__(image_dir=image_dir, debug=debug)

        self.classifier = svm.SVC(kernel=kernel, degree=degree, probability=True)

    def train(self):
        training_feats, training_labels = self.training_data()

        print("Fitting classifier...")
        self.classifier.fit(training_feats, training_labels)
        print("Fitted classifier.")

    def eval(self):
        eval_feats, eval_labels = self.all_data()

        print("Evaluating data points with classifier...")
        predictions = self.classifier.predict_proba(eval_feats)
        print("Evaluated data points with classifier.")

        return predictions

