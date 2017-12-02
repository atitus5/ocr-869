import math
import sys
sys.path.append("./")

import numpy as np

from utils.belief_prop import bp_error_correction
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
std_dev = 0.2
gaussian_noise = np.random.normal(mean, std_dev, predictions.shape)
noisy_predictions = np.add(predictions, gaussian_noise)
print("Done generating Gaussian noise.")

# Compute character error rate and word error rate before error correction
print("PRE-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(np.argmax(noisy_predictions, axis=1), kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(np.argmax(noisy_predictions, axis=1), kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))


# Run belief propagation with the bigram model
# Note: prediction is label vector, not one-hot matrix
bp_predictions = bp_error_correction(kjv, noisy_predictions)

# Compute character error rate and word error rate after error correction
print("POST-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(bp_predictions, kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(bp_predictions, kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Completed run!")
