import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Input, Flatten, Dense, GRU
from conformer_tf import ConformerConvModule, ConformerBlock
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
    
# tokenizer 로드
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# num_classes 설정
num_classes = len(tokenizer.word_index) + 1

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(output_dir):
    X = []
    y = []

    segment_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    segment_files.sort()

    for segment_file in segment_files:
        mfcc_filename = os.path.join(output_dir, segment_file)
        label_filename = mfcc_filename.replace('_mfcc.npy', '_labels.json')

        # MFCC 데이터 로드
        mfcc_data = np.load(mfcc_filename)

        # MFCC 데이터 차원 변환 (20, 26) -> )(26, 20
        mfcc_data = tf.transpose(mfcc_data, (1, 0))
    
        # 텐서로 변환
        mfcc_tensor = tf.convert_to_tensor(mfcc_data, dtype=tf.float32)

        # 라벨 데이터 로드
        with open(label_filename, 'r') as file:
            label_data = json.load(file)['labels']
        
        X.append(mfcc_tensor.numpy())
        y.append(label_data)

    X = tf.pad(X, [[0, 0], [0, 14], [0, 0]])  # (갯수, 26, 20) -> (갯수, 40, 20)
    print(np.array(X).shape, np.array(y))
    return np.array(X), np.array(y)

# 데이터 로드 및 전처리
directory_path = 'segments'
X, y = load_and_preprocess_data(directory_path)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 라벨 데이터 원-핫 인코딩 및 패딩
y_train_encoded = [to_categorical(label, num_classes=num_classes) for label in y_train]
y_test_encoded = [to_categorical(label, num_classes=num_classes) for label in y_test]

# 데이터셋 형태 조정
y_train = np.array(y_train)
y_test = np.array(y_test)

print("y_train shape:", np.array(y_train).shape)
print("y_test shape:", np.array(y_test).shape)

y_train_padded = pad_sequences(y_train_encoded, maxlen=40, padding='post', dtype='float32')
y_test_padded = pad_sequences(y_test_encoded, maxlen=40, padding='post', dtype='float32')

# 모델 아키텍처 정의
input_shape = X_train.shape[1:]  # MFCC 데이터의 형태
print("input_shape: ",input_shape)
inputs = Input(shape=input_shape)
print("inputs: ",inputs)

# # 인코더 부분: Conformer 블록과 모듈을 사용
conv_module = ConformerConvModule(
    dim=20,
    causal=False,
    expansion_factor=2,
    kernel_size=31,
    dropout=0.0,
)

conv_output = conv_module(inputs) + inputs
print("conv_output.shape: ",conv_output.shape)

conformer_block = ConformerBlock(
    dim=20,
    dim_head=64,
    heads=8,
    ff_mult=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    attn_dropout=0.0,
    ff_dropout=0.0,
    conv_dropout=0.0,
)
conformer_output = conformer_block(conv_output)
print("conformer_output.shape: ",conformer_output.shape)



# 디코더 부분: GRU 또는 LSTM 레이어 사용
decoder_units = 128  # 디코더의 유닛 수
decoder = GRU(decoder_units, return_sequences=True)(conformer_output)
decoder_output = GRU(decoder_units, return_sequences=True)(conformer_output)
print("decoder_output.shape: ", decoder_output.shape)

# 출력 레이어: 텍스트 라벨의 어휘 크기에 해당하는 유닛과 softmax 활성화 함수 사용
outputs = Dense(num_classes, activation='softmax')(decoder)
print("outputs: ",outputs)




# 모델 클래스 생성
model = Model(inputs=inputs, outputs=outputs)

# 모델 컴파일: 시퀀스 예측에 적합한 손실 함수와 메트릭 사용
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train_padded, batch_size=1, epochs=10, validation_split=0.2)

# 모델 평가
eval_results = model.evaluate(X_test, y_test_padded)