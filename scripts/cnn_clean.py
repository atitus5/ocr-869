import math
import sys
sys.path.append("./")

import numpy as np

from models.dnn import OCRCNN

from utils.belief_prop import bp_error_correction
from utils.viterbi import viterbi_error_correction
from utils.kjv_text import KJVTextDataset
from utils.metrics import char_err_rate, word_err_rate, confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


kjv = KJVTextDataset()

# Predict characters with convolutional neural net
kernel_sizes = [5, 3]
unit_counts = [220, 220]
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
preds = model.eval()

#low_values_flags = preds < .5  # Where values are low
#preds[low_values_flags] = 0  # All low values set to 0

print(preds.shape)
predictions = np.zeros(preds.shape)
for i in range(len(preds)):
    extra = i%(32*32)
    base = i-extra
    row = extra//32
    column = extra%32
    base+= (row+column*32)
    predictions[base] = preds[i]




print("Done predicting character labels using CNN.")

# Compute character error rate and word error rate before error correction
print("PRE-ERROR CORRECTION")
print("Computing character error rate (CER)...")
cer = char_err_rate(predictions, kjv)


CM = confusion_matrix(predictions, kjv)
print(CM)
ax = sns.heatmap(CM,
                 cmap="jet",
                 xticklabels=sorted(kjv.char_to_int.keys()),
                 yticklabels=sorted(kjv.char_to_int.keys()))
ax.set_title("Confusion Matrix")
plt.xlabel("Character 2")
plt.ylabel("Character 1")
plt.show()


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
