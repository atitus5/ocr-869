import math
import sys
sys.path.append("./")

import numpy as np

from utils.belief_prop import bp_error_correction
from utils.viterbi import viterbi_error_correction
from utils.kjv_text import KJVTextDataset
from utils.metrics import char_err_rate, word_err_rate

kjv = KJVTextDataset()

# Simply use ground truth one-hot vectors as predictions
# Just a baseline model -- not much accomplished here in general
predictions = kjv.one_hot_eval()

# Compute character error rate and word error rate before error correction
print("PRE-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(np.argmax(predictions, axis=1), kjv.one_hot_eval())
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(np.argmax(predictions, axis=1), kjv.one_hot_eval())
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Running belief prop with clean one-hot vectors...")

# Run belief propagation with the bigram model
# Note: prediction is label vector, not one-hot matrix
bp_predictions = bp_error_correction(kjv, predictions)

# Compute character error rate and word error rate after error correction
print("POST-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(bp_predictions, kjv.one_hot_eval())
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(bp_predictions, kjv.one_hot_eval())
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Completed BP run!")

print("Running Viterbi algorithm with clean one-hot vectors...")

# Run Viterbi algorithm with the bigram model
# Note: prediction is label vector, not one-hot matrix
viterbi_predictions = viterbi_error_correction(kjv, predictions)

# Compute character error rate and word error rate after error correction
print("POST-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(viterbi_predictions, kjv.one_hot_eval())
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(viterbi_predictions, kjv.one_hot_eval())
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Completed Viterbi run!")
