import math
import os
import sys
sys.path.append("./")

import numpy as np
from PIL import ImageFont
from sklearn.neighbors import KNeighborsClassifier
from skimage import io

from utils.kjv_text import KJVTextDataset

kjv = KJVTextDataset()
text_str = kjv.full_text

class OCRKNN(object):
    def __init__(self, image_dir="images/", n_neighbors=5):
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        # See scripts/generate_images.py
        self.font_size_in = 0.25
        self.font_size_pt = int(self.font_size_in * 72.0)
        self.font_path = "/Library/Fonts/Andale Mono.ttf"    # Specific to Mac OS -- change if needed
        self.font = ImageFont.truetype(self.font_path, self.font_size_pt)
        self.char_height, self.char_width = self.font.getsize("A")[0:2]
        self.chars_per_line = 32
        self.lines_per_img = 32
        self.image_dims_px = (self.char_height * self.chars_per_line,
                              (self.font_size_pt + 3) * self.lines_per_img)

        self.labels = kjv.image_label_mat(self.chars_per_line, self.lines_per_img)
        self.image_paths = [os.path.join(image_dir, filename) for filename in filter(lambda x: x.endswith(".png"), os.listdir(image_dir))]

    def train(self):
        print("Preparing training data...")

        # Samples are flattened individual character images
        char_image_size = (self.char_height, (self.font_size_pt + 3))
        flattened_size = char_image_size[0] * char_image_size[1]
        training_feats = np.empty((len(self.image_paths) * (self.chars_per_line * self.lines_per_img),
                                  flattened_size), dtype=float)

        # Flatten labels
        training_labels = self.labels.reshape((-1))
                                      
        training_indices = kjv.dataset_indices("train", self.chars_per_line, self.lines_per_img)
        for idx in training_indices:
            img = io.imread(self.image_paths[idx], as_grey=True)
            for x in range(self.lines_per_img):
                for y in range(self.chars_per_line):
                    feats = img[x * (self.font_size_pt + 3):(x + 1) * (self.font_size_pt + 3),
                                y * self.char_height:(y + 1) * self.char_height]
                    feats_flattened = feats.reshape((-1))
                    
                    feat_idx = (idx * flattened_size) + (x * self.chars_per_line) + y 
                    training_feats[feat_idx, :] = feats_flattened
        
        print("Prepared training data.")

        print("Fitting classifier...")
        self.classifier.fit(training_feats, training_labels)
        print("Fitted classifier.")

    def eval(self):
        print("Preparing evaluation data...")

        # Samples are flattened individual character images
        char_image_size = (self.char_height, (self.font_size_pt + 3))
        flattened_size = char_image_size[0] * char_image_size[1]
        eval_feats = np.empty((len(self.image_paths) * (self.chars_per_line * self.lines_per_img),
                                  flattened_size), dtype=float)
                                      
        # eval_indices = kjv.dataset_indices("eval", self.chars_per_line, self.lines_per_img)
        eval_indices = list(range(len(self.image_paths)))
        for idx in eval_indices:
            img = io.imread(self.image_paths[idx], as_grey=True)
            for x in range(self.lines_per_img):
                for y in range(self.chars_per_line):
                    feats = img[x * (self.font_size_pt + 3):(x + 1) * (self.font_size_pt + 3),
                                y * self.char_height:(y + 1) * self.char_height]
                    feats_flattened = feats.reshape((-1))
                    
                    feat_idx = (idx * flattened_size) + (x * self.chars_per_line) + y 
                    eval_feats[feat_idx, :] = feats_flattened
        
        print("Prepared eval data.")

        print("Evaluating data points with classifier...")
        predictions = self.classifier.predict_proba(eval_feats)
        print("Evaluated data points with classifier.")

        return predictions

