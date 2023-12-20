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
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras import layers, regularizers
import nltk

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




# 21-1. 모델 불러오기 (필요한 경우)
with tf.keras.utils.custom_object_scope({'PositionalEncoding': PositionalEncoding, 
                                         'TransformerEncoder': TransformerEncoder,
                                         'TransformerDecoder': TransformerDecoder}):
    model = tf.keras.models.load_model('transformer_dialect_to_standard.h5')
    
# 21-2. test_df 불러오기
test_df = pd.read_csv('test_data.csv')

# 21-3. 훈련 데이터 토크나이저 로드
with open('tokenizer_dialect.pkl', 'rb') as handle:
    tokenizer_dialect = pickle.load(handle)
with open('tokenizer_standard.pkl', 'rb') as handle:
    tokenizer_standard = pickle.load(handle)

# 21-4. 패딩 길이 적용을 위한, 최대 길이 매개변수 로드
with open('model_params.json', 'r') as json_file:
    params = json.load(json_file)
    max_length_dialect = params['max_length_dialect']
    max_length_standard = params['max_length_standard']
    
    
    
# # (+) 테스트 데이터에서 OOV 토큰이 존재하는 문장과 그 숫자 시퀀스를 찾아 출력
# oov_sentences_and_sequences = []
# for sentence in test_df['dialect']:
#     tokens = tokenizer_dialect.texts_to_sequences([sentence])[0] # 단일 문장을 숫자 시퀀스로 변환
#     # 각 토큰이 OOV 토큰인지 확인
#     if any(token == 1 for token in tokens):
#         oov_sentences_and_sequences.append((sentence, tokens))

# # (+) OOV 토큰이 존재하는 문장과 변환된 숫자 시퀀스를 출력
# for sentence, sequence in oov_sentences_and_sequences:
#     print("OOV 토큰이 포함된 방언 문장:", sentence)
#     print("해당 문장의 숫자 시퀀스:", sequence)

# oov_sentence_indexes = []
# # (+) OOV 토큰이 포함된 문장들의 인덱스를 탐색 후 출력 및 저장
# for index, sentence in enumerate(test_df['dialect']):
#     tokens = tokenizer_dialect.texts_to_sequences([sentence])[0]
#     if any(token == 1 for token in tokens):  # OOV 토큰을 확인
#         oov_sentence_indexes.append(index)
# print("OOV 문장 인덱스:", oov_sentence_indexes)



# 22. 테스트 데이터 전처리 (시퀀스 생성 -> 패딩) 
print("테스트 방언데이터 토큰화 전 결과:", test_df['dialect'].iloc[0])

# 22-1. 이때, 테스트 데이터는 이미 생성된 학습 토크나이저를 사용하며 숫자 시퀀스로 변환
test_dialect_seq = tokenizer_dialect.texts_to_sequences(test_df['dialect'])
print("테스트 방언데이터 토큰화 후 결과:", test_dialect_seq[0])

# 22-2. 테스트 데이터 패딩 적용
padded_test_dialect = pad_sequences(test_dialect_seq, maxlen=max_length_dialect, padding='post')
print("테스트 방언데이터 패딩 결과:", padded_test_dialect[0])

# 22-3. standard도 동일하게 시퀀스 -> 패딩 적용 (테스트 데이터의 stnadard는 원-핫 인코딩 할 필요 없음!)
test_standard_seq = tokenizer_standard.texts_to_sequences(test_df['standard'])
padded_test_standard = pad_sequences(test_standard_seq, maxlen=max_length_standard, padding='post')


model.summary()


# 23. 테스트 데이터에 대한 예측
def predict_sequence(model, input_seq, max_length_standard, tokenizer_standard, max_decoded_length):
    target_seq = np.zeros((1, max_length_standard))
    target_seq[0, 0] = tokenizer_standard.word_index['sos']

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens = model.predict([input_seq, target_seq])

        token_probabilities = output_tokens[0, -1, :]
        token_probabilities[0:4] = 0  # OOV, sos, eos 토큰의 확률을 0으로 설정
        token_probabilities /= token_probabilities.sum()  # 확률 정규화
        sampled_token_index = np.random.choice(np.arange(token_probabilities.size), p=token_probabilities)
        sampled_word = tokenizer_standard.index_word.get(sampled_token_index, '?')

        print("현재 디코더 입력 시퀀스:", target_seq[0, :len(decoded_sentence)+1])
        print("모델의 예측 확률 분포:", token_probabilities)
        print(f"예측된 인덱스: {sampled_token_index}, 예측된 단어: {sampled_word}")
        
        if sampled_word == 'eos' or len(decoded_sentence) >= max_decoded_length:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)
            if len(decoded_sentence) < max_length_standard - 1:
                target_seq[0, len(decoded_sentence)] = sampled_token_index

    return ' '.join(decoded_sentence)

# 테스트 데이터에 대한 예측 및 디버깅
for i in range(len(test_df)):
    input_seq = padded_test_dialect[i:i+1]
    max_decoded_length = len(test_df['dialect'].iloc[i].split())  # 원본 방언 문장의 단어 수로 최대 길이 설정
    predicted_sentence = predict_sequence(model, input_seq, max_length_standard, tokenizer_standard, max_decoded_length)
    print(f"원본 방언 문장: {test_df['dialect'].iloc[i]}")
    print(f"예측된 표준어 문장: {predicted_sentence}\n")




# 25. BLEU 스코어 계산 (NLP에 자주 사용되는 지표) // 0 ~ 1, 높을수록 유사성이 높다.
bleu_scores = []
for i in range(len(test_df)):
    input_seq = padded_test_dialect[i:i+1]
    decoded_sentence = decode_sequence(predicted_sequences[i], tokenizer_standard)
    reference = [test_df['standard'].iloc[i].split()]
    candidate = decoded_sentence.split()
    score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    bleu_scores.append(score)
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print("BLEU Score:", average_bleu_score)   