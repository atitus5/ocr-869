import math
import sys
sys.path.append("./")

import numpy as np

from utils.belief_prop import run_belief_prop
from utils.kjv_text import KJVTextDataset
from utils.metrics import char_err_rate, word_err_rate

print("Running belief prop with clean one-hot vectors...")

kjv = KJVTextDataset()

# Simply use ground truth one-hot vectors as predictions
# Just a baseline model -- not much accomplished here in general
predictions = kjv.one_hot()

# Run belief propagation with the bigram model
# Note: prediction is label vector, not one-hot matrix
# bp_predictions = run_belief_prop(kjv.char_bigram_matrix(), predictions)

# Add in backoff to keep probabilities relatively localized (think exponential moving avg)
char_dist_1pct = 1
backoff_alpha = math.pow(0.01, (1.0 / float(char_dist_1pct)))
print("Using backoff alpha %.6f (1%% contrib at %d char distance)" % (backoff_alpha, char_dist_1pct))
bp_predictions = run_belief_prop(kjv.char_bigram_matrix(), predictions, backoff_alpha=backoff_alpha)


# Compute character error rate and word error rate
print("Computing character error rate (CER)...")
cer = char_err_rate(bp_predictions, kjv)
print("Character error rate (CER): %.3f%%" % (cer * 100.0))

print("Computing word error rate (WER)...")
wer = word_err_rate(bp_predictions, kjv)
print("Word error rate (WER): %.3f%%" % (wer * 100.0))

print("Completed run!")
