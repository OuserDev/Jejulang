import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# CSV 파일을 불러오기
csv_file_path = 'dataset/Sample/label/dialect_standard_word_pairs.csv'
df = pd.read_csv(csv_file_path)

# 토크나이저 생성
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['dialect'].tolist() + df['standard'].tolist())

# tokenizer.word_index와 tokenizer.index_word의 일관성을 확인하는 코드
discrepancy_found = False
for word, index in tokenizer.word_index.items():
    retrieved_word = tokenizer.index_word.get(index)
    if retrieved_word != word:
        print(f"Discrepancy found: {word} -> {index} -> {retrieved_word}")
        discrepancy_found = True

if not discrepancy_found:
    print("All mappings between word_index and index_word are consistent.")


# 각 문장을 시퀀스로 변환
dialect_seq = tokenizer.texts_to_sequences(df['dialect'])
standard_seq = tokenizer.texts_to_sequences(df['standard'])

# 패딩 적용
dialect_padded = pad_sequences(dialect_seq, maxlen=1, padding='pre')
standard_padded = pad_sequences(standard_seq, maxlen=1, padding='post')

# 이미 생성된 토크나이저의 단어 인덱스를 사용하여 총 단어 수를 계산
vocab_size = len(tokenizer.word_index) + 1  # +1은 0 패딩을 위함

# 임베딩 차원 설정
embedding_dim = 100

# 원-핫 인코딩 대신에 표준어 문장을 사용하여 목표 데이터를 범주형 형태로 변환
num_classes = vocab_size
standard_padded_one_hot = to_categorical(standard_padded, num_classes=num_classes)

# LSTM 모델 구성 변경
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=1))
model.add(GRU(units=128, return_sequences=True))  # GRU 레이어 사용
model.add(Dropout(0.3))  # 드롭아웃 비율 증가
model.add(BatchNormalization())  # 배치 정규화 레이어 추가
model.add(GRU(units=64))  # 두 번째 GRU 레이어
model.add(Dense(64, activation='relu'))  # 추가 Dense 레이어 및 활성화 함수 변경
model.add(Dense(vocab_size, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 요약 정보 출력
model.summary()

# 데이터 준비
X = dialect_padded  # 입력 데이터
y = to_categorical(standard_padded, num_classes=vocab_size)  # 목표 데이터

# 학습 및 검증 세트로 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 에포크 수와 배치 크기 설정
epochs = 30
batch_size = 32

# 조기 종료를 위한 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 모델 훈련
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 훈련 결과 시각화
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_training_history(history)


# 예측하려는 방언 텍스트
dialect_text = "여기에 방언 그 아방에 모둠벌초 텍스트를 입력하세요."

words = dialect_text.split()
for word in words:
    seq = tokenizer.texts_to_sequences([word])
    
    # 시퀀스가 비어 있지 않은 경우에만 처리
    if len(seq[0]) > 0:
        padded_sequence = pad_sequences(seq, maxlen=1)
        prediction = model.predict(padded_sequence)
        predicted_index = np.argmax(prediction, axis=-1)[0]
        predicted_word = tokenizer.index_word.get(predicted_index)
        print(f"Predicted standard for '{word}': {predicted_word}")
    else:
        print(f"Word '{word}' not found in tokenizer index.")