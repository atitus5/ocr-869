from nltk import word_tokenize
import numpy as np

def char_err_rate(bp_predictions, kjv_text):
    # Convert belief prop predictions to char predictions
    bp_onehot = np.argmax(bp_predictions, axis=1)
    truth_onehot = np.argmax(kjv_text.one_hot(), axis=1)
    
    correct_chars = np.sum(bp_onehot == truth_onehot)
    total_chars = len(kjv_text.full_text)
    
    return (correct_chars / float(total_chars))

def word_err_rate(bp_predictions, kjv_text):
    # Convert belief prop predictions to sentence
    bp_onehot = np.argmax(bp_predictions, axis=1)
    bp_sentence = [kjv.int_to_char[x] for x in bp_onehot].join("")
    correct_tokens = 0

    # Compute WER as percentage of correct tokens as aligned by ground truth word tokens
    truth_tokens = word_tokenize(kjv.full_text)
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
            while char_idx != next_truth_token[0]:
                char_idx += 1
    
    return (correct_tokens / float(total_tokens))
