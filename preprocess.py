import json
import pandas as pd
import os
import re

def clean_and_split_sentence(sentence):
    """
    문장의 앞뒤 공백을 제거하고, 연속된 공백을 하나로 처리하여 어절로 분리하는 함수.
    """
    sentence = sentence.strip() # 앞뒤 공백 제거
    sentence = re.sub(r'\s+', ' ', sentence) # 연속된 공백을 하나로 처리
    return sentence.split()

def extract_dialect_standard(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    dialect_standard_pairs = []
    for segment in data['transcription']['segments']:
        if 'dialect' in segment and 'standard' in segment:
            dialect_standard_pairs.append((segment['dialect'], segment['standard']))
    return dialect_standard_pairs

def split_sentence_to_words(dialect_sentence, standard_sentence):
    if standard_sentence is None:
        standard_sentence = dialect_sentence
        print(f"None 그대로 넣음: {dialect_sentence} -> {standard_sentence}")
    dialect_words = clean_and_split_sentence(dialect_sentence)
    standard_words = clean_and_split_sentence(standard_sentence)
    if len(dialect_words) != len(standard_words):
        print(f"문장형 그대로 넣음: {dialect_sentence}, {standard_sentence}")
        return [(dialect_sentence, standard_sentence)]
    return list(zip(dialect_words, standard_words))

def preprocess_and_split(df):
    all_word_pairs = []
    for _, row in df.iterrows():
        word_pairs = split_sentence_to_words(row['dialect'], row['standard'])
        all_word_pairs.extend(word_pairs)
    word_df = pd.DataFrame(all_word_pairs, columns=['dialect', 'standard'])
    duplicated_pairs = word_df[word_df.duplicated(subset=['dialect'], keep=False)]
    for _, pair in duplicated_pairs.iterrows():
        print(f"중복되어 제거하였습니다.: {pair['dialect']} - {pair['standard']}")
    word_df.drop_duplicates(subset=['dialect'], keep='first', inplace=True)
    return word_df

def process_folder(folder_path):
    all_pairs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            pairs = extract_dialect_standard(file_path)
            all_pairs.extend(pairs)
    df = pd.DataFrame(all_pairs, columns=['dialect', 'standard'])
    df = preprocess_and_split(df)
    csv_file_path = os.path.join(folder_path, 'dialect_standard_word_pairs.csv')
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    return csv_file_path

folder_path = 'dataset/Sample/label'
csv_file_path = process_folder(folder_path)
print(f"CSV 파일이 저장된 경로: {csv_file_path}")

