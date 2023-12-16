import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CSV 파일을 불러오기
csv_file_path = 'dataset/Sample/label/dialect_standard_word_pairs.csv'  # 실제 파일 경로로 수정해야 합니다.
df = pd.read_csv(csv_file_path)

# 토크나이저 생성
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['dialect'].tolist() + df['standard'].tolist())

# 각 문장을 시퀀스로 변환
dialect_seq = tokenizer.texts_to_sequences(df['dialect'])
standard_seq = tokenizer.texts_to_sequences(df['standard'])

# 패딩 길이 결정
max_len_dialect = max(len(s) for s in dialect_seq)
max_len_standard = max(len(s) for s in standard_seq)
max_len = max(max_len_dialect, max_len_standard)

# 패딩 적용
dialect_padded = pad_sequences(dialect_seq, maxlen=max_len, padding='pre')
standard_padded = pad_sequences(standard_seq, maxlen=max_len, padding='post')

# 가장 긴 시퀀스의 인덱스를 찾기
longest_dialect_idx = np.argmax([len(s) for s in dialect_seq])
longest_standard_idx = np.argmax([len(s) for s in standard_seq])

# 가장 긴 시퀀스와 패딩 상태 출력
print("dialect_padded: ", dialect_padded)
print("standard_padded: ", standard_padded)
longest_dialect = df.iloc[longest_dialect_idx]['dialect']
longest_standard = df.iloc[longest_standard_idx]['standard']
print(f"The longest dialect sequence is: '{longest_dialect}' with length {max_len_dialect}")
print(f"Padded sequence: {dialect_padded[longest_dialect_idx]}")
print(f"The longest standard sequence is: '{longest_standard}' with length {max_len_standard}")
print(f"Padded sequence: {standard_padded[longest_standard_idx]}")

# 랜덤으로 5개 샘플을 뽑아서 패딩 상태와 원래 단어 이름을 출력
sample_indices = np.random.choice(len(df), size=5, replace=False)
for i in sample_indices:
    print(f"Original dialect: '{df.iloc[i]['dialect']}' - Padded sequence: {dialect_padded[i]}")
    print(f"Original standard: '{df.iloc[i]['standard']}' - Padded sequence: {standard_padded[i]}")
