import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense
from conformer_tf import ConformerConvModule, ConformerBlock
import numpy as np
import os

# .npy 파일 경로 설정
directory_path = 'segments'

# 디렉토리 내의 모든 .npy 파일을 리스트에 저장
mfcc_files = [file for file in os.listdir(directory_path) if file.endswith('.npy')]

# 정렬하여 순차적으로 처리하도록 하기 (옵션)
mfcc_files.sort()

# 모듈과 블록을 생성
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

# 모든 MFCC .npy 파일을 로드하고 모델에 입력
for file_name in mfcc_files:
    file_path = os.path.join(directory_path, file_name)
    mfcc_data = np.load(file_path)

    # 데이터 형태 출력
    print("MFCC data shape:", mfcc_data.shape)

    # 차원 변환 및 배치 차원 추가
    mfcc_data = np.expand_dims(mfcc_data, axis=0)  # (1, 20, 26)

    # Keras 모델을 사용하기 전에 텐서로 변환
    mfcc_data = tf.convert_to_tensor(mfcc_data, dtype=tf.float32)

    # Convolutional Module을 통과시킨 후 결과
    conv_output = conv_module(mfcc_data)

    # Conformer Block을 통과시킨 후 결과
    conformer_output = conformer_block(conv_output)

    # 모델의 출력 확인
    print(conformer_output.shape)

