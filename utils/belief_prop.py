import sys
import time

import numpy as np

def run_belief_prop(char_bigram_matrix, predictions):
    print("Running belief propagation")
    start_t = time.time()

    # Message_{i,j,k} is message from node i to node j (with dimension k = # unique chars)
    num_nodes, num_chars = predictions.shape[0:2]
    inc_msgs = np.zeros((num_nodes - 1, num_chars))     # Index i is message from i to (i + 1)
    dec_msgs = np.zeros((num_nodes - 1, num_chars))     # Index i is message from (i + 1) to i

    # BELIEF PROP
    # Compute edge conditions, normalizing in process
    inc_msgs[0, :] = np.matmul(char_bigram_matrix, predictions[0,:])
    inc_msgs[0, :] /= float(sum(inc_msgs[0, :]))
    dec_msgs[num_nodes - 2, :] = np.matmul(np.transpose(char_bigram_matrix), predictions[num_nodes - 1, :])
    dec_msgs[num_nodes - 2, :] /= float(sum(dec_msgs[num_nodes - 2, :]))

    # Compute all remaining messages. Operates bidirectionally.
    current_inc_msg = 1
    current_dec_msg = num_nodes - 3
    print_interval = int((num_nodes - 2) / 100)
    for i in range(num_nodes - 2):
        '''
        if i % print_interval == 0:
            # Print update in place
            sys.stdout.write("\rBelief propagation %d%% complete" % int((i / float(num_nodes - 2) * 100.0)))
            sys.stdout.flush()
        '''

        # Compute message in increasing direction, normalizing in process
        inc_msgs[current_inc_msg, :] = np.matmul(char_bigram_matrix,
                                                 np.multiply(inc_msgs[current_inc_msg - 1, :],
                                                             predictions[current_inc_msg, :]))
        inc_msgs[current_inc_msg, :] /= float(sum(inc_msgs[current_inc_msg, :]))
        current_inc_msg += 1

        # Compute message in decreasing direction, normalizing in process
        dec_msgs[current_dec_msg, :] = np.matmul(np.transpose(char_bigram_matrix),
                                                 np.multiply(dec_msgs[current_dec_msg + 1, :],
                                                             predictions[current_dec_msg + 1, :]))
        dec_msgs[current_dec_msg, :] /= float(sum(dec_msgs[current_dec_msg, :]))
        current_dec_msg -= 1
    
    # Insert newline to reset in-place update timer
    sys.stdout.write("\rBelief propagation 100% complete!\n")
    sys.stdout.flush()
    
    # Compute final marginal probabilities by multiplying incoming messages together
    # Uses labels instead of one-hot due to memory constraints
    final_predictions = np.zeros(num_nodes)

    # First node; edge case
    final_predictions[0] = np.argmax(dec_msgs[0, :])

    # Normal nodes
    for idx in range(1, num_nodes - 1):
        final_predictions[idx] = np.argmax(np.multiply(inc_msgs[idx - 1, :], dec_msgs[idx, :]))

    # Last node; edge case
    final_predictions[num_nodes - 1] = np.argmax(inc_msgs[num_nodes - 2, :])

    end_t = time.time()
    print("Ran belief prop in %.3f seconds" % (end_t - start_t)) 
    return final_predictions
