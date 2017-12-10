from nltk import word_tokenize
import numpy as np

def char_err_rate(bp_predictions, truth_one_hot):
    truth_labels = np.argmax(truth_one_hot, axis=1)
    correct_chars = np.sum(bp_predictions == truth_labels)
    total_chars = len(kjv_text.full_text)
    
    return (1.0 - (correct_chars / float(total_chars)))

def word_err_rate(bp_predictions, truth_one_hot):
    # Convert belief prop predictions to sentence
    bp_sentence = "".join([kjv_text.int_to_char[int(x)] for x in bp_predictions])
    correct_tokens = 0

    # Compute WER as percentage of correct tokens as aligned by ground truth word tokens
    truth_tokens = word_tokenize(kjv_text.full_text)
    total_tokens = len(truth_tokens)

    token_idx = 0
    char_idx = 0
    while token_idx < total_tokens:
        truth_token = truth_tokens[token_idx] 
        bp_token = bp_sentence[char_idx:char_idx + len(truth_token)]
        if truth_token == bp_token:
            correct_tokens += 1

        # Only worry about start character index of next token if not at end
        char_idx += len(truth_token)
        if token_idx < total_tokens - 1:
            next_truth_token = truth_tokens[token_idx + 1] 
            while kjv_text.full_text[char_idx] != next_truth_token[0]:
                char_idx += 1
        token_idx += 1
    
    return (1.0 - (correct_tokens / float(total_tokens)))
