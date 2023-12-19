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





## 폴더 내의 모든 json 파일을 처리 (process_json_files_to_csv로부터 호출)
def list_json_files(folder_path):
    return [file for file in os.listdir(folder_path) if file.endswith('.json')]

### 특수문자, 기호, _ 제거 (extract_sentences로부터 호출)
def clean_text(text):
    return re.sub(r'[\W_]+', ' ', text)

## json으로부터 dialect, standard를 추출하여 튜플로 반환 (process_json_files_to_csv로부터 호출)
def extract_sentences(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    sentences = []
    for sentence in data['transcription']['sentences']:
        dialect = clean_text(sentence['dialect'])
        standard = 'sos ' + clean_text(sentence['standard']) + ' eos'
        sentences.append((dialect, standard))
    
    return sentences

# 폴더 내 모든 json을 로드 -> 추출 -> 하나의 CSV로 통합
def process_json_files_to_csv(folder_path, output_csv_path):
    json_files = list_json_files(folder_path)
    all_sentences = []

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        sentences = extract_sentences(json_path)
        all_sentences.extend(sentences)
    
    df = pd.DataFrame(all_sentences, columns=['dialect', 'standard'])
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')





# 전처리 01. 데이터 로드 후 CSV화 (시작점)
folder_path = 'dataset/Sample/label'
output_csv_path = 'dialect_standard_sentences.csv'
process_json_files_to_csv(folder_path, output_csv_path)

# 02. CSV 로드하여 각 열의 문자열마다 불필요한 공백 제거 후 DF화
df = pd.read_csv(output_csv_path)
df['dialect'] = df['dialect'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
df['standard'] = df['standard'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
"""
print(df.head)
dialect  standard
0   아방 모녀 죽어부난 우리 어멍 질레 늙어쭈마씸    sos 아빠 먼저 죽어버려서 우리 어머니가 빨리 늙었지요 eos
1   우리 오라방 하간 거 다 알메 우리 오라방신디 들어 봐     sos 우리 오라버니 온갖 거 다 알아 우리 오라버니에게 물어 봐 eos
"""

# 03-1. 데이터 분할
train_df, test_df = train_test_split(df, test_size=0.1, random_state=14)
# 03-2. test_df를 CSV 파일로 저장
test_df.to_csv('test_data.csv', index=False)


# 04-1. 토크나이저 초기화 후, 훈련 데이터만 토크나이저 학습
oov_token = "OOV"
tokenizer_dialect = Tokenizer(oov_token=oov_token)
tokenizer_standard = Tokenizer(oov_token=oov_token, filters='')
tokenizer_dialect.fit_on_texts(train_df['dialect'])
tokenizer_standard.fit_on_texts(train_df['standard'])
"""
print(tokenizer_standard.word_index)
(6319, 2) {'OOV': 1, 'sos': 2, 'eos': 3, '이': 4, '때': 5, '다': 6, ... '너런': 2927}
"""
  
# 04-2. 훈련 데이터 토크나이저 저장 (시퀀스 생성 및 패딩 적용하지 않고 저장)
tokenizer_dialect_file = 'tokenizer_dialect.pkl'
tokenizer_standard_file = 'tokenizer_standard.pkl'
with open(tokenizer_dialect_file, 'wb') as handle:
    pickle.dump(tokenizer_dialect, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(tokenizer_standard_file, 'wb') as handle:
    pickle.dump(tokenizer_standard, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 05. 토큰화된 각 단어의 숫자를 기반으로, 문장의 숫자 시퀀스 생성 (= 문장 내 단어를 숫자로 변환)
sequences_dialect_train = tokenizer_dialect.texts_to_sequences(train_df['dialect'])
sequences_standard_train = tokenizer_standard.texts_to_sequences(train_df['standard'])
"""
print(sequences_dialect_train)
[[2, 574, 2511, 318, 988, 575], [1076, 2512, 13, 772, 55, 289, 2513, 39, 669, 2514], ... ]
"""
for seq in sequences_standard_train[:5]:
    print("시퀀스:", seq)
    if seq[-1] == tokenizer_standard.word_index['eos']:
        print("  'eos' 토큰이 마지막에 포함됨")
    else:
        print("  'eos' 토큰이 마지막에 없음")
        
# 06-1. 최대 길이를 구하여 패딩 적용
max_length_dialect = max(len(s) for s in sequences_dialect_train)
padded_sequences_dialect_train = pad_sequences(sequences_dialect_train, maxlen=max_length_dialect, padding='post')
max_length_standard = max(len(s) for s in sequences_standard_train) + 1  # 'eos' 토큰을 위한 추가 공간 포함
padded_sequences_standard_train = pad_sequences(sequences_standard_train, maxlen=max_length_standard, padding='post')
"""
print(padded_sequences_dialect_train)
[[   2  574 2511 ...    0    0    0], [1076 2512   13 ...    0    0    0], ... ]
"""

# 06-2. 추후 최대 길이 활용을 위한 저장
params = {
    'max_length_dialect': max_length_dialect,
    'max_length_standard': max_length_standard
}
with open('model_params.json', 'w') as json_file:
    json.dump(params, json_file)

# 07. 어휘 사전 크기 (단어의 수) 설정. Embedding layer에서 활용함 (패딩을 위한 0까지 포함하여 + 1)
vocab_size_dialect = len(tokenizer_dialect.word_index) + 1
vocab_size_standard = len(tokenizer_standard.word_index) + 1

# 08. 출력 데이터 (표준어 시퀀스)에 대하여 원-핫 인코딩
def one_hot_sequences(sequences, num_classes):
    one_hot_outputs = np.zeros((len(sequences), max_length_standard - 1, num_classes), dtype='int32')
    for i, sequence in enumerate(sequences):
        for t, word_index in enumerate(sequence):
            if t < max_length_standard - 1:  # 배열 범위 내 확인
                one_hot_outputs[i, t, word_index] = 1
    return one_hot_outputs

# 디코더 입력과 출력을 위한 데이터 준비
sequences_standard_decoder_input = [[2] + s[:-1] for s in sequences_standard_train]  # 'sos'에서 시작하여 'eos' 전까지
# 디코더 입력의 패딩
padded_sequences_standard_decoder_input = pad_sequences(sequences_standard_decoder_input, maxlen=max_length_standard - 1, padding='post')
one_hot_targets = one_hot_sequences(padded_sequences_standard_train, vocab_size_standard)
print("원-핫 인코딩된 데이터 예시:", one_hot_targets[0])
# 디코더 입력 시퀀스 중 일부를 출력
for seq in padded_sequences_standard_decoder_input[:5]:
    print("디코더 입력 시퀀스:", seq)
    if seq[0] == tokenizer_standard.word_index['sos']:
        print("  'sos' 토큰으로 시작함")
    else:
        print("  'sos' 토큰으로 시작하지 않음")
        
# 11. 모델 설정
learning_rate = 0.0001  # 학습률
embed_dim = 256         # 임베딩 차원
num_heads = 4           # 헤드 수
dense_dim = 512         # Dense 레이어 차원
num_layers = 3          # 레이어 수 조정

# 12. 인코더 설계
encoder_inputs = Input(shape=(None,))
x = Embedding(vocab_size_dialect, embed_dim)(encoder_inputs)
x = PositionalEncoding(embed_dim)(x)
for _ in range(num_layers):
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
encoder_outputs = x

# 13. 디코더 설계
decoder_inputs = Input(shape=(None,))
y = Embedding(vocab_size_standard, embed_dim)(decoder_inputs)
y = PositionalEncoding(embed_dim)(y)
for _ in range(num_layers):
    y = TransformerDecoder(embed_dim, dense_dim, num_heads)(y, encoder_outputs)
decoder_outputs = Dense(vocab_size_standard, activation='softmax')(y)

# 14. 모델 컴파일
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 15. 콜백함수(EarlyStopping) 설정
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
# 16. 모델 학습
history = model.fit(
    [padded_sequences_dialect_train, padded_sequences_standard_decoder_input],
    one_hot_targets,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# 17. 모델 저장
model.save('transformer_dialect_to_standard.h5')

