import os
import re
import json
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
import pickle

# 폴더 내 JSON 파일 목록을 가져오는 함수
def list_json_files(folder_path):
    return [file for file in os.listdir(folder_path) if file.endswith('.json')]

# 문장에서 특수문자를 제거하는 함수
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

# JSON 파일에서 dialect와 standard를 추출하는 함수
def extract_sentences(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    sentences = []
    for sentence in data['transcription']['sentences']:
        # 특수문자와 구두점을 제거
        dialect = clean_text(sentence['dialect'])
        standard = clean_text(sentence['standard'])
        sentences.append((dialect, standard))
    
    return sentences

# 모든 JSON 파일을 처리하고 결과를 CSV 파일로 저장하는 함수
def process_json_files_to_csv(folder_path, output_csv_path):
    json_files = list_json_files(folder_path)
    all_sentences = []

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        sentences = extract_sentences(json_path)
        all_sentences.extend(sentences)
    
    # DataFrame 생성
    df = pd.DataFrame(all_sentences, columns=['dialect', 'standard'])
    # CSV 파일로 저장
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

# 경로 설정
folder_path = 'dataset/Sample/label'  # JSON 파일들이 있는 폴더 경로
output_csv_path = 'dataset/Sample/label/dialect_standard_sentences.csv'  # 저장할 CSV 파일 경로

# 처리 실행
process_json_files_to_csv(folder_path, output_csv_path)

# CSV 파일 로드
df = pd.read_csv(output_csv_path)

# 데이터 검사 및 텍스트 정제 작업 수행
df['dialect'] = df['dialect'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
df['standard'] = df['standard'].str.strip().str.replace(r'[^\w\s]', '', regex=True)

# 토큰화 및 정수 인덱싱 (방언)
tokenizer_dialect = Tokenizer()
tokenizer_dialect.fit_on_texts(df['dialect'])
sequences_dialect = tokenizer_dialect.texts_to_sequences(df['dialect'])
max_length_dialect = max(len(s) for s in sequences_dialect)
padded_sequences_dialect = pad_sequences(sequences_dialect, maxlen=max_length_dialect, padding='post')

# 표준어 데이터에 대한 전처리 시작
# 표준어 데이터에 시작과 종료 토큰 추가
df['standard'] = '<sos> ' + df['standard'] + ' <eos>'

# 토큰화 및 정수 인덱싱 (표준어)
tokenizer_standard = Tokenizer()
tokenizer_standard.fit_on_texts(df['standard'])

# 토큰화 및 패딩 (표준어)
sequences_standard = tokenizer_standard.texts_to_sequences(df['standard'])
max_length_standard = max(len(s) for s in sequences_standard)
padded_sequences_standard = pad_sequences(sequences_standard, maxlen=max_length_standard, padding='post')
    
# 원-핫 인코딩 (타겟 데이터)
vocab_size_standard = len(tokenizer_standard.word_index) + 1
decoder_target_data = np.zeros((len(sequences_standard), max_length_standard, vocab_size_standard), dtype='float32')

for i, sequence in enumerate(sequences_standard):
    for t, word_index in enumerate(sequence):
        if word_index > 0:  # 0은 패딩을 나타내므로 제외
            decoder_target_data[i, t, word_index] = 1.

# 토크나이저 저장
with open('tokenizer_standard.pickle', 'wb') as handle:
    pickle.dump(tokenizer_standard, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('tokenizer_dialect.pickle', 'wb') as handle:
    pickle.dump(tokenizer_dialect, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('max_length_standard.txt', 'w') as file:
    file.write(str(max_length_standard))

with open('max_length_dialect.txt', 'w') as file:
    file.write(str(max_length_dialect))
    
# 모델 파라미터 설정
embedding_dim = 256
lstm_units = 512

# 두 어휘 사전 중 큰 크기를 선택
vocab_size = max(len(tokenizer_dialect.word_index), len(tokenizer_standard.word_index)) + 1
    
# 입력 시퀀스의 정의 및 처리
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# 디코더의 정의 및 처리
decoder_inputs = Input(shape=(None,))
decoder_embedding_layer = Embedding(vocab_size_standard, embedding_dim)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_standard, activation='softmax')(decoder_outputs)


# 모델 재정의 및 컴파일
model = Model([encoder_inputs, decoder_inputs], decoder_dense)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit([padded_sequences_dialect, padded_sequences_standard], decoder_target_data, batch_size=64, epochs=1, validation_split=0.2)

# 모델 저장
model.save('dialect_to_standard_translation_model.h5')

