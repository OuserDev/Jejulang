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

# Module - 지역적 특성을 포착하는데 중점을 두는, 컨포터 아키텍처의 개별 구성 요소
conv_module = ConformerConvModule(
    dim=512,
    causal=False,  # auto-regressive 여부
    expansion_factor=2,  # depthwise convolution을 위한 차원 확장 배수
    kernel_size=31,  # 커널 크기
    dropout=0.0,  # 드롭아웃 비율
)

# Block - 컨포머 모듈을 포함하는더 큰 단위, 복합적인 구조
conformer_block = ConformerBlock(
    dim=512,  # 차원 크기
    dim_head=64,  # 어텐션 헤드의 차원
    heads=8,  # 어텐션 헤드 수
    ff_mult=4,  # 피드 포워드 멀티플라이어
    conv_expansion_factor=2,  # 컨볼루션 확장 배수
    conv_kernel_size=31,  # 컨볼루션 커널 크기
    attn_dropout=0.0,  # 어텐션 드롭아웃 비율
    ff_dropout=0.0,  # 피드 포워드 드롭아웃 비율
    conv_dropout=0.0,  # 컨볼루션 드롭아웃 비율
)
    
# 모든 MFCC .npy 파일을 로드하고 모델에 입력
for file_name in mfcc_files:
    file_path = os.path.join(directory_path, file_name)
    mfcc_data = np.load(file_path)

    # 데이터 형태 출력
    print("MFCC data shape:", mfcc_data.shape)

    # 차원 변환 및 배치 차원 추가
    # 여기서 데이터의 마지막 차원을 512로 맞춰야 합니다.
    mfcc_data = np.expand_dims(mfcc_data, axis=0)  # (1, 20, 26)
    mfcc_data = np.expand_dims(mfcc_data, axis=-1)  # (1, 20, 26, 1)

    # Keras 모델을 사용하기 전에 텐서로 변환
    mfcc_data = tf.convert_to_tensor(mfcc_data, dtype=tf.float32)

    # Dense 레이어를 사용하여 차원 확장
    # (batch_size, sequence_length, feature_dim) 형태로 맞추기 위해 feature_dim을 512로 확장합니다.
    mfcc_data = Dense(512)(mfcc_data)

    # Convolutional Module을 통과시킨 후 결과
    conv_output = conv_module(mfcc_data) + mfcc_data

    # Conformer Block을 통과시킨 후 결과
    conformer_output = conformer_block(conv_output)

    # 모델의 출력 확인
    print(conformer_output.shape)
