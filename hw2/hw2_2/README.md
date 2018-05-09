# Scripts:
- 4 of all:
  - preprocess.py: training data preprocessing and dictionary prepraration
  - model_seq2seq.py: Attention + beam search seq2seq model implementation
  - inference.py: seq2seq model inference
  - hw2_seq2seq.sh: excecuting inference.py

# Requirements:  
- Python 3.5  
- tensorflow 1.6.0 

# Files:
> All required files should be named as follows and be in the same folder as scripts:
- word_dict.pkl

# Descriptions of scripts:
- model.py:
  - Execution:
  ```
    python3 preprocess.py
  ```
  - Output:
  ```
    Generate X_train.npy(previous sentence), Y_train.npy(next sentence), and word_dict.pkl(word_to_id and id_to_word).
  ```
- model_seq2seq.py:  
  - Execution:  
  ```
    python3 model_seq2seq.py
  ```
  - Output:  
  ```
    Save the seq2seq model into format(.ckpt).
  ```
- hw2_seq2seq.sh:  
  - Execution:  
  ```
    ./hw2_seq2seq.sh input.txt output.txt 
  ```
  - Output:  
  ```
    Generate sequences of sentences according to the input file in output.txt.
  ```
