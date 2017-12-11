from nltk import word_tokenize
import numpy as np

def char_err_rate(bp_predictions, kjv):
    truth_labels = np.argmax(kjv.one_hot(), axis=1)[:len(bp_predictions)]
    if len(bp_predictions.shape) > 1:
        bp_predictions = np.argmax(bp_predictions, axis=1)
    correct_chars = np.sum(np.equal(bp_predictions , truth_labels))
    total_chars = len(truth_labels)

    print("%d correct characters out of %d" % (correct_chars, total_chars))
    
    return (1.0 - (correct_chars / float(total_chars)))

def confusion_matrix(bp_predictions, kjv):
    if len(bp_predictions.shape) > 1:
        bp_predictions = np.argmax(bp_predictions, axis=1)
    truth_labels = np.argmax(kjv.one_hot(), axis=1)[:len(bp_predictions)]
    matrix = np.zeros((kjv.unique_chars(), kjv.unique_chars()))
    for i in range(len(truth_labels)):
        matrix[truth_labels[i]][int(bp_predictions[i])]+=1
    return matrix/matrix.sum(axis=1, keepdims=True) 

def word_err_rate(bp_predictions, kjv):
    if len(bp_predictions.shape) > 1:
        bp_predictions = np.argmax(bp_predictions, axis=1)
    # Convert belief prop predictions to sentence
    bp_sentence = "".join([kjv.int_to_char[int(x)] for x in bp_predictions])
    correct_tokens = 0

    # Compute WER as percentage of correct tokens as aligned by ground truth word tokens
    truth_tokens = word_tokenize(kjv.full_text[:len(bp_predictions)])
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
            while kjv.full_text[char_idx] != next_truth_token[0]:
                char_idx += 1
        token_idx += 1
    
    return (1.0 - (correct_tokens / float(total_tokens)))
