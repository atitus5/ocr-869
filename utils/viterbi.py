import sys
import time

import numpy as np

def run_viterbi(char_bigram_matrix, predictions, backoff_alpha=1.0):
    print("Running Viterbi algorithm...")
    start_t = time.time()

    # We have (num_nodes) states
    # Use our (num_chars x num_chars) bigram matrix as state transition probabilities
    num_nodes, num_chars = predictions.shape[0:2]

    # Convert to log probabilities to avoid underflow after many multiplications
    log_predictions = np.log(np.clip(predictions, np.finfo(float).eps, 1.0))    # Avoid log(0)
    char_log_bigram_matrix = np.log(char_bigram_matrix)

    # Initialize tables
    best_path_logprobs = np.ones((num_chars, num_nodes), dtype=np.float32) * np.NINF
    best_path_logprobs[:, 0] = log_predictions[0, :]    # Initial state probabilities

    last_char = np.zeros((num_chars, num_nodes), dtype=np.int32)
    last_char[:, 0] = -1                                # Indicates start of sequence

    # Fill in tables (forward pass)
    print_interval = int((num_nodes - 1) / 100)
    for char_idx in range(1, num_nodes):
        if (char_idx - 1) % print_interval == 0:
            # Print update in place
            sys.stdout.write("\rViterbi forward pass %d%% complete" % int(((char_idx - 1) / float(num_nodes - 1) * 100.0)))
            sys.stdout.flush()

        path_logprobs = np.add(char_log_bigram_matrix,
                               np.matmul(best_path_logprobs[:, char_idx - 1].reshape((num_chars, 1)),
                                         np.ones((1, num_chars))))
        best_path_logprobs[:, char_idx] = np.amax(path_logprobs, axis=0)
        last_char[:, char_idx] = np.argmax(path_logprobs, axis=0)
    
    # Insert newline to reset in-place update timer
    sys.stdout.write("\rViterbi forward pass 100% complete!\n")
    sys.stdout.flush()

    # Backtrack to get most likely char sequence (backward pass)
    print("Viterbi backward pass...")
    final_predictions = np.zeros(num_nodes, dtype=np.int32)
    final_predictions[num_nodes - 1] = np.argmax(best_path_logprobs[:, num_nodes - 1])
    for char_idx in range(num_nodes - 2, -1, -1):
        final_predictions[char_idx] = last_char[final_predictions[char_idx + 1], char_idx + 1]
    print("Viterbi backward pass complete!")

    end_t = time.time()
    print("Ran Viterbi algorithm in %.3f seconds" % (end_t - start_t)) 
    return final_predictions
