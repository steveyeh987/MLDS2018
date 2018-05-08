
# coding: utf-8

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import pandas as pd
import collections
import itertools
import pickle
import re

x_train = []
y_train = []

temp = []

with open('clr_conversation.txt', 'r') as f:
    for sentence in f.readlines():
        if sentence.startswith('+'):
            x_train.extend(temp[:-1])
            y_train.extend(temp[1:])
            temp = []
        else:
            sentence = sentence.strip('\n').split()
            temp.append(list(sentence))

BUFFER_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

## get the vocaboluary 
list_of_all_words = [w for sent in x_train for w in sent]
counter = collections.Counter(list_of_all_words)
counter_dict = {k:v for k, v in counter.items()}
vocab = sorted(counter_dict.items(), key=lambda x:-x[1])[:50000] # top 50000 words

## create word_to_idx, and idx_to_word
vocab = [i[0] for i in vocab]

word_to_idx = {}
idx_to_word = {}
# add in BUFFER_TOKENS
for i in range(len(BUFFER_TOKENS)):
    idx_to_word[int(i)] = BUFFER_TOKENS[i]
    word_to_idx[BUFFER_TOKENS[i]] = i

for i in range(len(vocab)):
    word_to_idx[vocab[i]] = i + len(BUFFER_TOKENS)
    idx_to_word[int(i + len(BUFFER_TOKENS))] = vocab[i]

word_dict = {}
word_dict['idx_to_word'] = idx_to_word
word_dict['word_to_idx'] = word_to_idx
vocab_size = len(word_to_idx)

with open("word_dict.pkl","wb") as f:
    pickle.dump(word_dict, f)

# convert sentences into encoding/integers
# pad all sentence to length of padding_len - 2 
def _convert_sentence_to_numbers(s):
    """Convert a sentence s (a list of words) to list of numbers using word_to_idx"""
    UNK_IDX = BUFFER_TOKENS.index('<UNK>')
    PAD_IDX = BUFFER_TOKENS.index('<PAD>')
    START_TOKEN = BUFFER_TOKENS.index('<BOS>')
    END_IDX = BUFFER_TOKENS.index('<EOS>')
    padding_len = 35
    s_encoded = [START_TOKEN]
    s_encoded += [word_to_idx.get(w) for w in s if w in word_to_idx]
    s_encoded += [END_IDX]
    s_encoded += [PAD_IDX] * (padding_len - len(s_encoded))
    return s_encoded

X_train = [_convert_sentence_to_numbers(s) for s in x_train]
Y_train = [_convert_sentence_to_numbers(s) for s in y_train]

np.save('X_train.npy', np.array(X_train))
np.save('Y_train.npy', np.array(Y_train))

