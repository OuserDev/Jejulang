import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Input, Flatten, Dense
from conformer_tf import ConformerConvModule, ConformerBlock
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json

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

        # MFCC 데이터 차원 변환 (20, 26) -> (26, 20)
        mfcc_data = np.transpose(mfcc_data, (1, 0))

        # MFCC 데이터에 batch_size 차원 추가 (1, 26, 20)
        mfcc_data = np.expand_dims(mfcc_data, axis=0)

        # 텐서로 변환
        mfcc_tensor = tf.convert_to_tensor(mfcc_data, dtype=tf.float32)

        # 라벨 데이터 로드
        with open(label_filename, 'r') as file:
            label_data = json.load(file)['labels']
        
        X.append(mfcc_tensor.numpy())
        y.append(label_data)

    print(np.array(X).shape, np.array(y).shape)
    return np.array(X), np.array(y)

# 데이터 로드 및 전처리
directory_path = 'segments'
X, y = load_and_preprocess_data(directory_path)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 아키텍처 정의
input_shape = X_train.shape[1:]  # MFCC 데이터의 형태
inputs = Input(shape=input_shape)

# 모듈과 블록 생성
conv_module = ConformerConvModule(
    dim=512,
    causal=False,
    expansion_factor=2,
    kernel_size=31,
    dropout=0.0,
)

conformer_block = ConformerBlock(
    dim=512,
    dim_head=64,
    heads=8,
    ff_mult=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    attn_dropout=0.0,
    ff_dropout=0.0,
    conv_dropout=0.0,
)

x = conv_module(inputs) + inputs
x = conformer_block(x)
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)  # 예측 라벨 수에 맞게 조정

model = Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# 모델 평가
eval_results = model.evaluate(X_test, y_test)