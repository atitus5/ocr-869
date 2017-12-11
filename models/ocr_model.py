import math
import os
import sys
sys.path.append("./")

import numpy as np
from PIL import ImageFont
from skimage import io

from utils.kjv_text import KJVTextDataset
import matplotlib

# Base class for other model variants
class OCRModel(object):
    def __init__(self, image_dir="images/", debug=False):
        self.kjv = KJVTextDataset()

        # See scripts/generate_images.py
        self.font_size_in = 0.25
        self.font_size_pt = int(self.font_size_in * 72.0)
        self.font_path = "utils/Andale-Mono.ttf"    # Specific to Mac OS -- change if needed
        self.font = ImageFont.truetype(self.font_path, self.font_size_pt)
        self.char_height, self.char_width = self.font.getsize("A")[0:2]
        self.chars_per_line = 32
        self.lines_per_img = 32
        self.image_dims_px = (self.char_height * self.chars_per_line,
                              (self.font_size_pt + 3) * self.lines_per_img)
        self.char_image_size = (self.char_height, (self.char_width + 3))

        # Sort NUMERICALLY, not LEXICOGRAPHICALLY... goodness
        self.labels = self.kjv.image_label_mat(self.chars_per_line, self.lines_per_img)
        self.image_paths = [os.path.join(image_dir, filename) for filename in sorted(filter(lambda x: x.endswith(".png"), os.listdir(image_dir)), key=lambda filename:int(filename.rstrip(".png")))]

        # Dynamically load these later
        self._all_data = None
        self._training_data = None
        self._val_data = None

        self.debug = debug

    def all_data(self):
        if self._all_data is None:
            print("Preparing all data...")

            # Samples are flattened individual character images
            flattened_size = self.char_image_size[0] * self.char_image_size[1]
            chars_per_image = self.chars_per_line * self.lines_per_img

            if self.debug:
                # Quick prototyping
                all_indices = list(range(10))
            else:
                all_indices = range(len(self.kjv.dataset_indices("train", self.chars_per_line, self.lines_per_img)) + len(self.kjv.dataset_indices("val", self.chars_per_line, self.lines_per_img)))

            all_feats = np.empty((len(all_indices) * chars_per_image,
                                       flattened_size), dtype=float)
            all_labels = np.zeros((len(all_indices) * chars_per_image), dtype=int)


            for i in range(len(all_indices)):
                all_idx = all_indices[i]
                img = io.imread(self.image_paths[all_idx], as_grey=True)
                for x in range(self.chars_per_line):
                    for y in range(self.lines_per_img):
                        feats = img[y * (self.char_width + 3):(y + 1) * (self.char_width + 3),
                                    x * self.char_height:(x + 1) * self.char_height]
                        #io.imshow(feats)
        
                        feats_flattened = feats.reshape((-1))
                        
                        feat_idx = (i * chars_per_image) + (x * self.lines_per_img) + y 
                        all_feats[feat_idx, :] = feats_flattened
                        all_labels[feat_idx] = self.labels[all_idx, (y * self.lines_per_img) + x]
                        #print(all_labels[feat_idx])
                        #matplotlib.pyplot.show()
            self._all_data = (all_feats, all_labels)
            
            print("Prepared all data.")

        return self._all_data

    def training_data(self):
        if self._training_data is None:
            print("Preparing training data...")

            # Samples are flattened individual character images
            flattened_size = self.char_image_size[0] * self.char_image_size[1]
            chars_per_image = self.chars_per_line * self.lines_per_img

            if self.debug:
                # Quick prototyping
                training_indices = list(range(9))
            else:
                training_indices = self.kjv.dataset_indices("train", self.chars_per_line, self.lines_per_img)

            training_feats = np.empty((len(training_indices) * chars_per_image,
                                       flattened_size), dtype=float)
            training_labels = np.zeros((len(training_indices) * chars_per_image), dtype=int)


            for i in range(len(training_indices)):
                training_idx = training_indices[i]
                img = io.imread(self.image_paths[training_idx], as_grey=True)
                for x in range(self.chars_per_line):
                    for y in range(self.lines_per_img):
                        feats = img[y * (self.char_width + 3):(y + 1) * (self.char_width + 3),
                                    x * self.char_height:(x + 1) * self.char_height]
                        #io.imshow(feats)
        
                        feats_flattened = feats.reshape((-1))
                        
                        feat_idx = (i * chars_per_image) + (x * self.lines_per_img) + y 
                        training_feats[feat_idx, :] = feats_flattened
                        training_labels[feat_idx] = self.labels[training_idx, (y * self.lines_per_img) + x]
                        #print(training_labels[feat_idx])
                        #matplotlib.pyplot.show()
            self._training_data = (training_feats, training_labels)
            
            print("Prepared training data.")

        return self._training_data

    def val_data(self):
        if self._val_data is None:
            print("Preparing val data...")

            # Samples are flattened individual character images
            flattened_size = self.char_image_size[0] * self.char_image_size[1]
            chars_per_image = self.chars_per_line * self.lines_per_img

            if self.debug:
                # Quick prototyping
                val_indices = list(range(9, 10))
            else:
                val_indices = self.kjv.dataset_indices("train", self.chars_per_line, self.lines_per_img)

            val_feats = np.empty((len(val_indices) * chars_per_image,
                                       flattened_size), dtype=float)
            val_labels = np.zeros((len(val_indices) * chars_per_image), dtype=int)

            for i in range(len(val_indices)):
                val_idx = val_indices[i]
                img = io.imread(self.image_paths[val_idx], as_grey=True)
                for x in range(self.lines_per_img):
                    for y in range(self.chars_per_line):
                        feats = img[y * (self.char_width + 3):(y + 1) * (self.char_width + 3),
                                    x * self.char_height:(x + 1) * self.char_height]
                        feats_flattened = feats.reshape((-1))
                        
                        feat_idx = (i * chars_per_image) + (x * self.lines_per_img) + y 
                        val_feats[feat_idx, :] = feats_flattened
                        val_labels[feat_idx] = self.labels[val_idx, (y * self.lines_per_img) + x]

            self._val_data = (val_feats, val_labels)
            
            print("Prepared val data.")

        return self._val_data
