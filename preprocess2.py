# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# 데이터 로드
csv_file_path = 'dataset/Sample/label/dialect_standard_word_pairs.csv'
df = pd.read_csv(csv_file_path)

# 토크나이저 생성 및 학습
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['dialect'].tolist() + df['standard'].tolist())

# 시퀀스 변환 및 패딩
dialect_seq = tokenizer.texts_to_sequences(df['dialect'])
standard_seq = tokenizer.texts_to_sequences(df['standard'])
dialect_padded = pad_sequences(dialect_seq, maxlen=1, padding='pre')
standard_padded = pad_sequences(standard_seq, maxlen=1, padding='post')

# 모델 구성
vocab_size = len(tokenizer.word_index) + 1
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=1))
model.add(GRU(units=128, return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(GRU(units=64))
model.add(Dense(64, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# 모델 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train, X_val, y_train, y_val = train_test_split(dialect_padded, to_categorical(standard_padded, num_classes=vocab_size), test_size=0.2, random_state=42)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 데이터 준비: 'standard' 단어 집합에 대해서만 모델을 학습합니다.
standard_sentences = [row.split() for row in df['standard'].tolist()]
# 'standard' 단어에 대한 Word2Vec 모델 학습
standard_embedding_model = Word2Vec(standard_sentences, vector_size=100, window=5, min_count=1, workers=4)

# 테스트 문장
dialect_text = "여기에 방언 그 아방에 모둠벌초 텍스트를 입력하세요."
words = dialect_text.split()

# 각 'dialect' 단어에 대해 가장 유사한 'standard' 단어 찾기
for word in words:
    # 'dialect' 단어가 'standard' 단어 임베딩 모델에 있는지 확인
    if word in standard_embedding_model.wv.key_to_index:
        similar_word = standard_embedding_model.wv.most_similar(word, topn=1)[0][0]
        print(f"Most similar standard word for '{word}': {similar_word}")
    else:
        print(f"'{word}' not found in 'standard' Word2Vec model.")