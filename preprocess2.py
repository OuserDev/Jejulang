from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np 

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

# 번역(예측)을 위한 함수 정의
def translate_sentence(input_sentence):
    # 입력 문장 전처리
    input_sequence = tokenizer_dialect.texts_to_sequences([input_sentence])
    input_padded = pad_sequences(input_sequence, maxlen=max_length_dialect, padding='post')
    
    # 인코더 문맥 벡터 생성
    encoder_context = model.layers[2](model.layers[1](input_padded))

    # 디코더 초기 입력(시작 토큰)
    target_seq = np.zeros((1, 1))
    sos_token_index = tokenizer_standard.word_index.get('<sos>', None)
    if sos_token_index is None:
        raise ValueError("Start-of-sentence token '<sos>' not found in tokenizer word index.")
    target_seq[0, 0] = sos_token_index

    # 번역 결과 저장
    translated_sentence = ''
    max_length_standard = max_length_standard

    while True:
        # 디코더에 현재 시퀀스 통과
        output_tokens, h, c = model.layers[4](model.layers[3](target_seq), initial_state=encoder_context)

        # 가장 확률 높은 단어 선택
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_standard.index_word[sampled_token_index]

        # 종료 토큰 확인
        if sampled_word == '<eos>' or len(translated_sentence) > max_length_standard:
            break

        # 결과에 단어 추가
        translated_sentence += ' ' + sampled_word

        # 타겟 시퀀스 업데이트
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # 상태 업데이트
        encoder_context = [h, c]

    return translated_sentence.strip()

# 사용 예시
input_sentence = "여기에 제주도 방언 문장 입력"
translated_sentence = translate_sentence(input_sentence)
print("번역된 문장:", translated_sentence)