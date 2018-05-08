
# coding: utf-8


import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from keras.preprocessing.text import text_to_word_sequence
from tensorflow.python.layers.core import Dense
from itertools import chain
import numpy as np
import pandas as pd
import collections
import itertools
import pickle
import sys
import re

X_test = []
input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, 'r') as f:
    for sentence in f.readlines():
        sentence = sentence.strip('\n').split()
        X_test.append(list(sentence))

BUFFER_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

with open("word_dict.pkl","rb") as f:
    word_dict = pickle.load(f)

word_to_idx = word_dict['word_to_idx']
vocab_size = len(word_dict['word_to_idx'])

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

X_test = [_convert_sentence_to_numbers(s) for s in X_test]
X_test = np.array(X_test)

batch_size = 100
layers = 2
max_len = 35
nm_epochs = 50
beam_width = 3
input_embedding_size = 128
encoder_hidden_units = 512
decoder_hidden_units = 512

tf.reset_default_graph()

encoder_inputs = tf.placeholder(shape=(None, max_len), dtype=tf.int32)
decoder_inputs = tf.placeholder(shape=(batch_size, max_len - 1), dtype=tf.int32)
decoder_targets = tf.placeholder(shape=(batch_size, max_len - 1), dtype=tf.int32)
input_seq_len = tf.placeholder(shape=(None,), dtype=tf.int32)
target_seq_len = tf.placeholder(shape=(None,), dtype=tf.int32)
sampling_prob = tf.placeholder(shape=(), dtype=tf.float32)

def next_batch(source, target, batch_size):
    # Shuffle data
    source = np.array(source)
    target = np.array(target)
    shuffle_indices = np.random.permutation(np.arange(len(target)))
    source = source[shuffle_indices]
    target = target[shuffle_indices]
    
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        input_seqlen_batch = [list(row).index(2) + 1 for row in source_batch]
        target_seqlen_batch = [list(row).index(2) for row in target_batch]

        yield np.array(source_batch), np.array(target_batch), np.array(input_seqlen_batch), np.array(target_seqlen_batch)

def range_with_end(start, stop, step):
    return chain(range(start, stop, step), (stop,))

def build_model(mode, batch_size, input_embedding_size, encoder_hidden_units, decoder_hidden_units):
    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    def make_cell(hidden_units):
        return tf.contrib.rnn.LSTMCell(hidden_units)

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        encoder_multi_gru = tf.nn.rnn_cell.MultiRNNCell([make_cell(encoder_hidden_units) for _ in range(layers)], state_is_tuple=True)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
            encoder_multi_gru, encoder_inputs_embedded, 
            sequence_length=input_seq_len, dtype=tf.float32
        )
        
        if mode == "training":
            # ATTENTION
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units = encoder_hidden_units, 
                memory = encoder_outputs,
                memory_sequence_length = input_seq_len)

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.nn.rnn_cell.MultiRNNCell([make_cell(encoder_hidden_units) for _ in range(layers)], state_is_tuple=True),
                attention_mechanism = attention_mechanism,
                attention_layer_size = encoder_hidden_units)
            
        elif mode == "testing":
            # BEAM SEARCH TILE
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
            Input_seq_len = tf.contrib.seq2seq.tile_batch(input_seq_len, multiplier=beam_width)
            encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=beam_width)

            # ATTENTION (PREDICTING)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units = encoder_hidden_units, 
                memory = encoder_outputs,
                memory_sequence_length = Input_seq_len)

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.nn.rnn_cell.MultiRNNCell([make_cell(encoder_hidden_units) for _ in range(layers)], state_is_tuple=True),
                attention_mechanism = attention_mechanism,
                attention_layer_size = encoder_hidden_units)
        
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        output_layer = Dense(vocab_size)
        
        if mode == "training":
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_inputs_embedded,
                target_seq_len,
                time_major = False)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_final_state),
                output_layer=output_layer)

            maximum_iterations = tf.reduce_max(target_seq_len)

            decoder_outputs, decoder_final_state, seq_len = tf.contrib.seq2seq.dynamic_decode(
                                                            decoder, output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=maximum_iterations)

            decoder_logits = tf.identity(decoder_outputs.rnn_output)
            
            return decoder_logits, maximum_iterations
            
        elif mode == "testing":
            batch_size = tf.shape(encoder_inputs)[0:1]
            start_tokens = tf.ones(batch_size, dtype=tf.int32)
            
            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = decoder_cell,
                embedding = embeddings,
                start_tokens = start_tokens,
                end_token = 2,
                initial_state=decoder_cell.zero_state(batch_size*beam_width, tf.float32).clone(cell_state=encoder_final_state),
                beam_width = beam_width,
                output_layer = output_layer,
                length_penalty_weight = 0.0)

            infer_dec_outputs, infer_dec_last_state, infer_seq_len = tf.contrib.seq2seq.dynamic_decode(
                inference_decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=2*max_len)

            predictions = tf.identity(infer_dec_outputs.predicted_ids[:, :, 0])
    
            return predictions

def train_neural_network(mode):
    final_preds = []
    predictions = build_model(mode, batch_size, input_embedding_size, encoder_hidden_units, decoder_hidden_units)
    
    saver = tf.train.Saver()
    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        saver.restore(sess, "lstm_model_attention.ckpt")

        # Testing
        sample_size = len(X_test)
        
        for batch_start, batch_end in zip(range_with_end(0, sample_size, batch_size), range_with_end(batch_size, sample_size, batch_size)):
            x_test = X_test[batch_start:batch_end]
            input_seqlen = [list(row).index(2) + 1 for row in x_test]
            pred = sess.run(predictions, feed_dict={encoder_inputs: x_test, 
                                                             input_seq_len: input_seqlen})
            final_preds.append(pred)

        
    return final_preds

final_preds = train_neural_network(mode="testing")
predictions = [lst[:list(lst).index(2)] if 2 in lst else lst for batch in final_preds for lst in batch] 
predictions = [[k for k, g in itertools.groupby(lst)] for lst in predictions] # remove consecutive duplicates
text = [[word_dict['idx_to_word'][_id] for _id in row] for row in predictions]

with open(output_path, 'w') as f:
    for t in (text):
        f.write('{}\n'.format(' '.join(t)))
