import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# 가정: `conformer_model`이라는 Conformer 모델이 이미 정의되어 있고,
# `train_dataset`과 `val_dataset`이 tf.data.Dataset 객체로 준비되어 있다.

# 옵티마이저 설정
optimizer = Adam(learning_rate=1e-4)

# 손실 함수 설정
loss_function = SparseCategoricalCrossentropy(from_logits=True)

# 성능 평가 지표 설정
train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
val_accuracy = SparseCategoricalAccuracy(name='val_accuracy')

# 훈련 스텝 함수 정의
@tf.function
def train_step(model, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss

# 검증 스텝 함수 정의
@tf.function
def val_step(model, inputs, labels):
    predictions = model(inputs, training=False)
    loss = loss_function(labels, predictions)
    val_accuracy.update_state(labels, predictions)
    return loss

# 훈련 루프
for epoch in range(epochs):
    # 훈련
    for inputs, labels in train_dataset:
        loss = train_step(conformer_model, inputs, labels)

    # 검증
    for inputs, labels in val_dataset:
        val_loss = val_step(conformer_model, inputs, labels)

    # 에포크마다 결과 출력
    template = "Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}"
    print(template.format(epoch+1,
                          loss,
                          train_accuracy.result() * 100,
                          val_loss,
                          val_accuracy.result() * 100))

    # 메트릭 리셋
    train_accuracy.reset_states()
    val_accuracy.reset_states()
