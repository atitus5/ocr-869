import time

import numpy as np

class BeliefPropRunner(object):
    def run(self, char_bigram_matrix, predictions):
        start_t = time.time()

        # TODO: change to actual BP
        final_predictions = predictions

        end_t = time.time()
        print("Ran belief prop in %.3f seconds" % (end_t - start_t))

        return final_predictions
