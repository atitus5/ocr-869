import math
import sys
import time

from nltk import word_tokenize
import numpy as np

def bp_error_correction(kjv, all_predictions):
    start_t = time.time()

    # Run belief propagation to correct any words not found in dictionary
    print("Setting up word set and tokenizing predictions...")
    word_set = set(word_tokenize(kjv.full_text))    
    predicted_char_ints = np.argmax(all_predictions, axis=1)
    predicted_chars = list(map(lambda x: kjv.int_to_char[x], predicted_char_ints))
    predicted_sentence = "".join(predicted_chars)
    predicted_tokens = word_tokenize(predicted_sentence)
    print("Done setting up.")
    
    # Add in backoff to keep probabilities relatively localized (think exponential moving avg)
    char_dist_1pct = 5  # Arbitrary; can be changed
    backoff_alpha = math.pow(0.01, (1.0 / float(char_dist_1pct)))
    print("Using backoff alpha %.6f (1%% contrib at %d char distance)" % (backoff_alpha, char_dist_1pct))

    # Correct only words that don't fall into our word set
    print("Correcting character errors with belief propagation...")
    corrected_predictions = predicted_char_ints
    token_idx = 0
    char_idx = 0
    print_interval = int(len(predicted_tokens) / 100)
    for token_idx in range(len(predicted_tokens)):
        if token_idx % print_interval == 0:
            # Print update in place
            sys.stdout.write("\rError correction %d%% complete" % int(token_idx / float(len(predicted_tokens) * 100.0)))
            sys.stdout.flush()

        token = predicted_tokens[token_idx]

        if token not in word_set:
            # Attempt to fix the error
            start = char_idx
            end = char_idx + len(token)
            new_char_predictions = run_belief_prop(kjv.char_bigram_matrix(),
                                                   all_predictions[start:end, :],
                                                   backoff_alpha=backoff_alpha)
            corrected_predictions[start:end] = new_char_predictions

        # Only worry about start character index of next token if not at end
        char_idx += len(token)
        if token_idx < len(predicted_tokens) - 1:
            next_token = predicted_tokens[token_idx + 1] 
            while predicted_sentence[char_idx] != next_token[0]:
                char_idx += 1

    # Insert newline to reset in-place update timer
    sys.stdout.write("\rError correction 100% complete!\n")
    sys.stdout.flush()
    
    end_t = time.time()
    print("Corrected errors with belief prop in %.3f seconds" % (end_t - start_t)) 
    return corrected_predictions
    

def run_belief_prop(char_bigram_matrix, predictions, backoff_alpha=1.0):
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
    for i in range(num_nodes - 2):
        # Compute message in increasing direction, normalizing in process
        inc_msgs[current_inc_msg, :] = np.matmul(char_bigram_matrix,
                                                 np.multiply(backoff_alpha * inc_msgs[current_inc_msg - 1, :],
                                                             predictions[current_inc_msg, :]))
        inc_msgs[current_inc_msg, :] /= float(sum(inc_msgs[current_inc_msg, :]))
        current_inc_msg += 1

        # Compute message in decreasing direction, normalizing in process
        dec_msgs[current_dec_msg, :] = np.matmul(np.transpose(char_bigram_matrix),
                                                 np.multiply(backoff_alpha * dec_msgs[current_dec_msg + 1, :],
                                                             predictions[current_dec_msg + 1, :]))
        dec_msgs[current_dec_msg, :] /= float(sum(dec_msgs[current_dec_msg, :]))
        current_dec_msg -= 1
    
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

    return final_predictions
