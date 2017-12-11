import math
import sys
sys.path.append("./")

import numpy as np

from models.cnn import OCRCNN

from utils.belief_prop import bp_error_correction
from utils.viterbi import viterbi_error_correction
from utils.kjv_text import KJVTextDataset
from utils.metrics import char_err_rate, word_err_rate

kjv = KJVTextDataset()

# Predict characters with convolutional neural net
kernel_sizes = [5, 3]
unit_counts = [128, 64]
strides = [1, 1]
maxpool_sizes = [0, 0]
print("Using kernels %s" % str(kernel_sizes))
print("Using unit counts %s" % str(unit_counts))
print("Using strides %s" % str(strides))
print("Using max-pool sizes %s" % str(maxpool_sizes))
model = OCRCNN(kernel_sizes=kernel_sizes, unit_counts=unit_counts, strides=strides, maxpool_sizes=maxpool_sizes)
# model = OCRCNN(kernel_sizes=kernel_sizes, unit_counts=unit_counts, strides=strides, maxpool_sizes=maxpool_sizes, debug=True)

print("Training CNN...")
model.train()
print("Done training CNN.")

print("Predicting character labels using CNN...")
predictions = model.eval()
print("Done predicting character labels using CNN.")

# Compute character error rate and word error rate before error correction
print("PRE-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(predictions, kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(predictions, kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Running belief prop...")

# Run belief propagation with the bigram model
# Note: prediction is label vector, not one-hot matrix
bp_predictions = bp_error_correction(kjv, predictions)

# Compute character error rate and word error rate after error correction
print("POST-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(bp_predictions, kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(bp_predictions, kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Completed BP run!")

print("Running Viterbi algorithm...")

# Run Viterbi algorithm with the bigram model
# Note: prediction is label vector, not one-hot matrix
viterbi_predictions = viterbi_error_correction(kjv, predictions)

# Compute character error rate and word error rate after error correction
print("POST-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(viterbi_predictions, kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(viterbi_predictions, kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Completed Viterbi run!")
