import math
import sys
sys.path.append("./")

import numpy as np

from models.svm import OCRSVM

from utils.belief_prop import bp_error_correction
from utils.viterbi import viterbi_error_correction
from utils.kjv_text import KJVTextDataset
from utils.metrics import char_err_rate, word_err_rate

kjv = KJVTextDataset()

# Predict characters with an SVM
# model = OCRSVM()
model = OCRSVM(debug=True)

print("Training SVM...")
model.train()
print("Done training SVM.")

print("Predicting character labels using SVM...")
predictions = model.eval()
print("Done predicting character labels using SVM.")

# Compute character error rate and word error rate before error correction
print("PRE-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(np.argmax(predictions, axis=1), kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(np.argmax(predictions, axis=1), kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Running belief prop ...")

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

print("Running Viterbi algorithm ...")

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
