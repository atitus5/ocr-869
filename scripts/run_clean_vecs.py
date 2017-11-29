import sys
sys.path.append("./")

import numpy as np

from utils.belief_prop import BeliefPropRunner
from utils.kjv_text import KJVTextDataset
from utils.metrics import char_err_rate, word_err_rate

print("Running with clean one-hot vectors...")

kjv = KJVTextDataset()
bp = BeliefPropRunner()

# Simply use ground truth one-hot vectors as predictions
# Just a baseline model -- not much accomplished here in general
predictions = kjv.one_hot()

# Run belief propagation with the bigram model
bp_predictions = bp.run(kjv.char_bigram_matrix(), predictions)

# Compute character error rate and word error rate
print("Computing character error rate (CER)...")
cer = char_err_rate(bp_predictions, kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(bp_predictions, kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Completed run!")
