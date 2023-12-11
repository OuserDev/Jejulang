import tensorflow as tf
from conformer_tf import ConformerConvModule, ConformerBlock

# Convolutional Module 생성
conv_module = ConformerConvModule(
    dim=512,
    causal=False,  # auto-regressive 여부
    expansion_factor=2,  # depthwise convolution을 위한 차원 확장 배수
    kernel_size=31,  # 커널 크기
    dropout=0.0,  # 드롭아웃 비율
)

# Conformer Block 생성
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

# 임의의 입력 데이터 생성
input_data = tf.random.normal([1, 1024, 512])

# Convolutional Module을 통과시킨 후 결과
conv_output = conv_module(input_data) + input_data

# Conformer Block을 통과시킨 후 결과
conformer_output = conformer_block(conv_output)

# 모델의 출력 확인
print(conformer_output.shape)
