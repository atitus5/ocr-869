import time

import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
from nltk import ngrams
import numpy as np
import seaborn as sns; sns.set()

class KJVTextDataset(object):
    def __init__(self):
        super(KJVTextDataset, self).__init__()

        # Ignore newlines
        self.full_text = gutenberg.raw('bible-kjv.txt').replace('\n', '')
        self.words = gutenberg.words('bible-kjv.txt')

        # Determine counts of unique chars in text, plus a mapping of char->int (for bigram matrix)
        self.char_counts = dict()
        self.char_to_int = dict()
        self.int_to_char = []
        for char in self.full_text:
            if char in self.char_counts:
                self.char_counts[char] += 1
            else:
                self.char_counts[char] = 1

            if char not in self.char_to_int:
                char_int = len(self.char_to_int)
                self.char_to_int[char] = char_int
                self.int_to_char.append(char)

        # Don't compute bigrams or one-hot in initialization -- do on-demand
        self.char_bigrams = None    # |unique chars| x |unique chars|
        self.one_hot_matrix = None

    def unique_chars(self):
        return len(self.char_counts)

    def char_count(self, char):
        if char in self.char_counts:
            return self.char_counts[char]
        else:
            return 0

    def char_bigram_score(self, char_1, char_2):
        if self.char_bigrams is None:
            # Need to compute these first!
            self._compute_char_bigrams()
        
        return self.char_bigrams[self.char_to_int[char_1],
                                 self.char_to_int[char_2]]

    def char_bigram_matrix(self):
        if self.char_bigrams is None:
            # Need to compute these first!
            self._compute_char_bigrams()
        
        return self.char_bigrams

    def char_bigram_heatmap(self):
        if self.char_bigrams is None:
            # Need to compute these first!
            self._compute_char_bigrams()
        
        # Visualize the char bigram matrix as a heatmap
        # Use log probabilities due to large spread from low to high
        ax = sns.heatmap(np.log(self.char_bigrams),
                         cmap="jet",
                         xticklabels=sorted(self.char_to_int.keys()),
                         yticklabels=sorted(self.char_to_int.keys()))
        ax.set_title("Char bigram log-probabilities")
        plt.xlabel("Character 1")
        plt.ylabel("Character 2")
        plt.show()

    def _compute_char_bigrams(self):
        print("Recomputing character bigrams...", flush=True)
        start_t = time.time()

        all_char_bigrams = ngrams(list(self.full_text), 2)

        # First compute raw counts
        # Uses eps instead of zero so log prob viz works
        self.char_bigrams = np.ones((self.unique_chars(), self.unique_chars())) * np.finfo(float).eps
        total_bigrams = 0
        for char_bigram in all_char_bigrams:
            char_1, char_2 = char_bigram[0:2]
            self.char_bigrams[self.char_to_int[char_1],
                              self.char_to_int[char_2]] += 1
            total_bigrams += 1

        # Now, normalize into probabilities
        self.char_bigrams /= float(total_bigrams)

        end_t = time.time()
        print("Done recomputing character bigrams (%.3f seconds)." % (end_t - start_t), flush=True)

    def one_hot(self):
        # Get one-hot ground truth vectors for each char in the text as a matrix
        if self.one_hot_matrix is None:
            self.one_hot_matrix = np.zeros((len(self.full_text), self.unique_chars()), dtype=float)
            for i in range(len(self.full_text)):
                one_hot_idx = self.char_to_int[self.full_text[i]]
                self.one_hot_matrix[i, one_hot_idx] = 1.0
        return self.one_hot_matrix
