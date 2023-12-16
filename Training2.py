import tensorflow as tf
from conformer_tf import ConformerConvModule
from conformer_tf import ConformerBlock

layer = ConformerConvModule(
    dim=20,
    causal=False,  # whether it is auto-regressive
    expansion_factor=2,  # what multiple of the dimension to expand for the depthwise convolution
    kernel_size=31,
    dropout=0.0,
)

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
mfcc_data = tf.random.normal([20, 26])
mfcc_data = tf.transpose(mfcc_data, (1, 0))  # (20, 26) -> (26, 20)
mfcc_data = tf.expand_dims(mfcc_data, axis=0)  # (26, 20) -> (1, 26, 20)
print("MFCC데이터 차원: ",mfcc_data.shape)

mfcc_data_padded = tf.pad(mfcc_data, [[0, 0], [0, 14], [0, 0]])  # (1, 26, 20) -> (1, 40, 20)
conformer_output = layer(mfcc_data) + mfcc_data_padded
print("ConformerConvModule 출력 차원: ", conformer_output.shape)

conformer_b_output = conformer_block(conformer_output)  # (1, 1024, 512)
print("ConformerConvModule 출력 차원: ", conformer_b_output)