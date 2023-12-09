import json
import os
from pydub import AudioSegment
import librosa
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 라벨링된 json 파일 로드
def load_metadata(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        metadata = json.load(file)
    return metadata

# 문자열 형태의 시간을 파싱하여, 밀리초 단위로 변환
def parse_time(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    return (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000 + int(time_obj.microsecond / 1000)

# 메타데이터에서 필요한 정보 추출 (대본)
def extract_information(metadata):
    transcription_info = metadata['transcription']
    return transcription_info

# 음성 분할 및 세그먼트로 분할
def segment_audio(audio_file, segments, output_dir):
    audio = AudioSegment.from_wav(audio_file)
    for i, segment in enumerate(segments):
        start_time = parse_time(segment['startTime'])
        end_time = parse_time(segment['endTime'])
        segment_audio = audio[start_time:end_time]
        segment_audio.export(os.path.join(output_dir, f'segment_{i}.wav'), format='wav')

# 각 세그먼트에서 MFCC 추출
def extract_mfcc(audio_file, n_mfcc=20):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# MFCC 데이터 정규화
def normalize_mfcc(mfcc):
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_normalized = (mfcc - mfcc_mean[:, np.newaxis]) / mfcc_std[:, np.newaxis]
    return mfcc_normalized

# 텍스트 라벨 처리 (추출 -> 토큰화 -> 패딩 적용)
def process_text_labels(metadata):
    standard_texts = [segment['standard'] for segment in metadata['transcription']['segments']] # 텍스트 라벨 추출
    tokenizer = Tokenizer() # 텍스트 라벨 토큰화, 각 토큰에 고유 정수 할당
    tokenizer.fit_on_texts(standard_texts)
    sequences = tokenizer.texts_to_sequences(standard_texts) 
    padded_sequences = pad_sequences(sequences, padding='post') # 모든 텍스트 라벨이 동일한 길이를 갖도록 패딩 추가
    return padded_sequences, tokenizer.word_index

# 메인 함수
def main(json_file, audio_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata = load_metadata(json_file)
    transcription_info = extract_information(metadata)
    segment_audio(audio_file, transcription_info['segments'], output_dir)

    # 텍스트 라벨 처리
    text_labels, word_index = process_text_labels(metadata)
    
    for i, segment_file in enumerate(os.listdir(output_dir)):
        if segment_file.endswith('.wav'):
            mfcc = extract_mfcc(os.path.join(output_dir, segment_file))
            mfcc_normalized = normalize_mfcc(mfcc)
            
            # MFCC와 텍스트 라벨 디버깅 출력
            print(f"Segment: {segment_file}")
            print(f"Normalized MFCC Shape: {mfcc_normalized.shape}")
            print("Normalized MFCC (First 5 frames):")
            print(mfcc_normalized[:, :5])
            if i < len(text_labels):
                print(f"Text Label (Segment {i}): {text_labels[i]}")
                print(f"Corresponding Words: {[word for word, index in word_index.items() if index in text_labels[i]]}")
            print("")
            # MFCC를 저장하거나 분석하는 로직 추가

# 실행
json_file = 'dataset/Sample/label/st_set1_collectorjj9_speakerjj14_0_7.json'
audio_file = 'dataset/Sample/voice/st_set1_collectorjj9_speakerjj14_0_7.wav'
output_dir = 'segments'  # 세그먼트 파일을 저장할 폴더
main(json_file, audio_file, output_dir)