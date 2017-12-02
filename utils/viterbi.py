import sys
import time

from nltk import word_tokenize
import numpy as np

def viterbi_error_correction(kjv, all_predictions):
    start_t = time.time()

    # Run Viterbi algorithm to correct any words not found in dictionary
    print("Setting up word set and tokenizing predictions...")
    word_set = set(word_tokenize(kjv.full_text))    
    predicted_char_ints = np.argmax(all_predictions, axis=1)
    predicted_chars = list(map(lambda x: kjv.int_to_char[x], predicted_char_ints))
    predicted_sentence = "".join(predicted_chars)
    predicted_tokens = word_tokenize(predicted_sentence)
    print("Done setting up.")
    
    # Correct only words that don't fall into our word set
    print("Correcting character errors with Viterbi algorithm...")
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
            new_char_predictions = run_viterbi(kjv.char_bigram_matrix(),
                                               all_predictions[start:end, :])
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
    print("Corrected errors with Viterbi algorithm in %.3f seconds" % (end_t - start_t)) 
    return corrected_predictions


def run_viterbi(char_bigram_matrix, predictions):
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
    for char_idx in range(1, num_nodes):
        path_logprobs = np.add(char_log_bigram_matrix,
                               np.matmul(best_path_logprobs[:, char_idx - 1].reshape((num_chars, 1)),
                                         np.ones((1, num_chars))))
        best_path_logprobs[:, char_idx] = np.amax(path_logprobs, axis=0)
        last_char[:, char_idx] = np.argmax(path_logprobs, axis=0)
    
    # Backtrack to get most likely char sequence (backward pass)
    final_predictions = np.zeros(num_nodes, dtype=np.int32)
    final_predictions[num_nodes - 1] = np.argmax(best_path_logprobs[:, num_nodes - 1])
    for char_idx in range(num_nodes - 2, -1, -1):
        final_predictions[char_idx] = last_char[final_predictions[char_idx + 1], char_idx + 1]

    return final_predictions
