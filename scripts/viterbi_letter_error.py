import math
import sys
sys.path.append("./")

import numpy as np
import random

from utils.viterbi import viterbi_error_correction
from utils.kjv_text import KJVTextDataset
from utils.metrics import char_err_rate, word_err_rate

print("Running belief prop with one-hot vectors degraded by Gaussian noise...")

kjv = KJVTextDataset()

# Simply use ground truth one-hot vectors as predictions
# Just a baseline model -- not much accomplished here in general
predictions = kjv.one_hot()

# Generate Gaussian noise (don't worry about normalization/rectification,
# the error correction will do this automatically later)
print("Generating Gaussian noise...")
mean = 0.0
std_dev = 0.1

noise_per_letter = np.identity(predictions.shape[1])
for i in range(predictions.shape[1]):
	rand = np.random.choice(range(1,4), 1, p=[.85, .1, .05])
	vector = np.ones(rand)
	noise_per_letter[i] = np.convolve(noise_per_letter[i], vector, "same")
	noise_per_letter[i]/=noise_per_letter[i].sum()



noisy_predictions = np.zeros(predictions.shape)
for i in range(noisy_predictions.shape[0]):
	noisy_predictions[i] = noise_per_letter[np.argmax(predictions[i])] + np.random.normal(mean, std_dev, predictions.shape[1])
print("Done generating Gaussian noise.")

# Compute character error rate and word error rate before error correction
print("PRE-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(np.argmax(noisy_predictions, axis=1), kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(np.argmax(noisy_predictions, axis=1), kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

# Run Viterbi algorithm with the bigram model
# Note: prediction is label vector, not one-hot matrix
viterbi_predictions = viterbi_error_correction(kjv, noisy_predictions)

# Compute character error rate and word error rate after error correction
print("POST-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(viterbi_predictions, kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(viterbi_predictions, kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Completed run!")
