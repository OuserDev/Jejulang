import os
import re
import json
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, max_len, d_model):
        angle_rads = self.get_angles(np.arange(max_len)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = layers.Dense(dense_dim, activation="relu",
                                       kernel_regularizer=regularizers.l2(1e-4))
        self.dense_output = layers.Dense(embed_dim,
                                         kernel_regularizer=regularizers.l2(1e-4))
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attention_output = self.attention(inputs, inputs)
        proj_input = self.norm1(inputs + self.dropout1(attention_output, training=training))
        proj_output = self.dense_proj(proj_input)
        return self.norm2(proj_input + self.dropout2(self.dense_output(proj_output), training=training))

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = layers.Dense(dense_dim, activation="relu",
                                       kernel_regularizer=regularizers.l2(1e-4))
        self.dense_output = layers.Dense(embed_dim,
                                         kernel_regularizer=regularizers.l2(1e-4))
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_outputs, training=False, mask=None):
        attention_output1 = self.attention1(inputs, inputs)
        norm1_output = self.norm1(inputs + self.dropout1(attention_output1, training=training))
        attention_output2 = self.attention2(norm1_output, encoder_outputs)
        norm2_output = self.norm2(norm1_output + self.dropout2(attention_output2, training=training))
        proj_output = self.dense_proj(norm2_output)
        return self.norm3(norm2_output + self.dropout3(self.dense_output(proj_output), training=training))


# 01. 모델 불러오기
with tf.keras.utils.custom_object_scope({'PositionalEncoding': PositionalEncoding, 
                                         'TransformerEncoder': TransformerEncoder,
                                         'TransformerDecoder': TransformerDecoder}):
    model = tf.keras.models.load_model('transformer_dialect_to_standard.h5')

# 02-1. 훈련 데이터 토크나이저 로드
with open('tokenizer_dialect.pkl', 'rb') as handle:
    tokenizer_dialect = pickle.load(handle)
with open('tokenizer_standard.pkl', 'rb') as handle:
    tokenizer_standard = pickle.load(handle)

# 02-2. 패딩 길이 적용을 위한, 최대 길이 매개변수 로드
with open('model_params.json', 'r') as json_file:
    params = json.load(json_file)
    max_length_dialect = params['max_length_dialect']
    max_length_standard = params['max_length_standard']




 
# 입력
input_sentence = "아치 때 도새기 잡는 날은 몸국헹덜 먹었지"

# 입력 데이터 전처리 - 시퀀스 변환
sequence = tokenizer_dialect.texts_to_sequences([input_sentence])
print("입력데이터 토큰화 후 결과:", sequence[0])
# 입력 데이터 전처리 - 패딩 적용
padded_sequence = pad_sequences(sequence, maxlen=max_length_dialect, padding='post')
print("입력데이터 패딩 결과:", padded_sequence)

print("토큰화된 단어들:", end=" ")
for word_index in sequence[0]:
    word = tokenizer_dialect.index_word.get(word_index, None)
    if word:
        print(f"{word}", end=" ")
    else:
        print(f"OOV", end=" ")
print()


# 인코더 입력 준비
encoder_input = padded_sequence

# 디코더 입력 초기화
decoder_input = np.zeros((1, max_length_standard), dtype='int32')
decoder_input[0, 0] = tokenizer_standard.word_index['sos']

# 모델을 사용하여 단계별로 시퀀스 예측
for i in range(1, max_length_standard):
    predictions = model.predict([encoder_input, decoder_input])[0, i - 1, :]
    sampled_token_index = np.argmax(predictions)
    decoder_input[0, i] = sampled_token_index
    if sampled_token_index == tokenizer_standard.word_index['eos']:
        break

# 예측된 숫자 시퀀스를 텍스트로 변환
predicted_sentence = ' '.join([tokenizer_standard.index_word[token] for token in decoder_input[0] if token > 0 and token != tokenizer_standard.word_index['sos']])
print("예측된 표준어 문장:", predicted_sentence)
