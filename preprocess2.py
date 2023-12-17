from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import pickle
import numpy as np 

# LSTM 유닛 수 설정
lstm_units = 512

# 토크나이저 로드
with open('tokenizer_dialect.pickle', 'rb') as handle:
    tokenizer_dialect = pickle.load(handle)

with open('tokenizer_standard.pickle', 'rb') as handle:
    tokenizer_standard = pickle.load(handle)

with open('max_length_standard.txt', 'r') as file:
    max_length_standard = int(file.read())

with open('max_length_dialect.txt', 'r') as file:
    max_length_dialect = int(file.read())

# 저장된 모델 로드
model = load_model('dialect_to_standard_translation_model.h5')

# 인코더 모델 정의
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # LSTM 레이어
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

# 디코더 모델 정의
decoder_embedding_layer = model.layers[3]
decoder_lstm = model.layers[5]
decoder_dense = model.layers[6]

decoder_inputs = model.input[1]  # 디코더의 입력
decoder_state_input_h = Input(shape=(lstm_units,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(lstm_units,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

print("Encoder Model Summary:")
encoder_model.summary()
print("\nDecoder Model Summary:")
decoder_model.summary()
# tokenizer_standard의 토큰 인덱스와 단어 사전 확인
print("\nTokenizer Standard Word Index:\n", tokenizer_standard.word_index)

# 번역(예측)을 위한 함수 정의
def translate_sentence(input_sentence):
    # 입력 문장 전처리
    input_sequence = tokenizer_dialect.texts_to_sequences([input_sentence])
    input_padded = pad_sequences(input_sequence, maxlen=max_length_dialect, padding='post')
    
    # 인코더 모델로 상태 얻기
    states_value = encoder_model.predict(input_padded)
    print("인코더 모델 상태 값:", states_value)  # 인코더 모델 상태 확인

    # 디코더의 초기 입력(시작 토큰)
    target_seq = np.array([[tokenizer_standard.word_index['sos']]])
    stop_condition = False
    translated_sentence = ''

    # 번역 과정
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 가장 확률 높은 단어 선택
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_standard.index_word[sampled_token_index]
        print(f"예측된 단어: {sampled_word} (인덱스: {sampled_token_index})")  # 예측된 단어 확인

        # 종료 토큰 확인
        if sampled_word == 'eos' or len(translated_sentence.split()) >= max_length_standard:
            stop_condition = True
        else:
            translated_sentence += ' ' + sampled_word
            # 타겟 시퀀스 및 상태 업데이트
            target_seq = np.array([[sampled_token_index]])
            states_value = [h, c]
            print(f"다음 입력 시퀀스: {target_seq}")  # 다음 입력 시퀀스 확인

    return translated_sentence.strip()

# 사용 예시
input_sentence = "여기에 제주도 방언 문장 입력"
translated_sentence = translate_sentence(input_sentence)
print("번역된 문장:", translated_sentence)