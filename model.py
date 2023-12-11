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
def segment_audio(audio_file, segments, output_dir, base_name):
    audio = AudioSegment.from_wav(audio_file)
    for i, segment in enumerate(segments):
        start_time = parse_time(segment['startTime'])
        end_time = parse_time(segment['endTime'])
        segment_audio = audio[start_time:end_time]
        segment_filename = f'{base_name}_segment_{i}.wav'  # 변경된 부분
        segment_audio.export(os.path.join(output_dir, segment_filename), format='wav')

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

# 전체 텍스트 데이터를 위한 word_index 생성
def create_word_index(json_files, label_dir):
    all_texts = []
    for json_file in json_files:
        json_path = os.path.join(label_dir, json_file)
        metadata = load_metadata(json_path)
        for segment in metadata['transcription']['segments']:
            # 'standard'가 null이면 'dialect' 값을 사용, 둘 다 null이면 빈 문자열 사용
            text = segment['standard'] if segment['standard'] is not None else (segment['dialect']  if segment['dialect'] is not None else "")
            all_texts.append(text)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    return tokenizer

# 메인 함수 수정
def main(dataset_dir, output_dir):
    label_dir = os.path.join(dataset_dir, 'label')
    voice_dir = os.path.join(dataset_dir, 'voice')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    audio_files = [f for f in os.listdir(voice_dir) if f.endswith('.wav')]
    
    # 전체 데이터를 위한 Tokenizer 생성
    tokenizer = create_word_index(json_files, label_dir)
    
    # 모든 json 파일에 대하여 처리를 반복한다.
    for json_file in json_files:
        segment_counter = 0
        base_name = json_file.split('.')[0]
        audio_file = f"{base_name}.wav"
        
        if audio_file in audio_files:
            json_path = os.path.join(label_dir, json_file)
            audio_path = os.path.join(voice_dir, audio_file)
            
            metadata = load_metadata(json_path)
            transcription_info = extract_information(metadata)
            segment_audio(audio_path, transcription_info['segments'], output_dir, base_name)
            
            # 여기서 각 세그먼트의 텍스트 라벨을 처리한다.
            standard_texts = [segment['standard'] if segment['standard'] is not None else (segment['dialect'] if segment['dialect'] is not None else "") for segment in transcription_info['segments']]
            sequences = tokenizer.texts_to_sequences(standard_texts)
            padded_sequences = pad_sequences(sequences, padding='post')
            
            index_word = tokenizer.index_word
            # 각 세그먼트 파일에 대해 MFCC 추출 및 정규화
            for segment_file in sorted(os.listdir(output_dir)):
                if segment_file.startswith(base_name) and segment_file.endswith('.wav'):
                    mfcc = extract_mfcc(os.path.join(output_dir, segment_file))
                    mfcc_normalized = normalize_mfcc(mfcc)
                    
                    # MFCC와 텍스트 라벨 디버깅 출력 (여기에 추가 로직 구현)
                    #print(f"Segment: {segment_file}")
                    print(f"Normalized MFCC Shape: {mfcc_normalized.shape}")
                    #print("Normalized MFCC (First 5 frames):")
                    #print(mfcc_normalized[:, :5])
                    if segment_counter < len(padded_sequences):
                        segment_labels = padded_sequences[segment_counter]
                        words = [index_word.get(idx, '') for idx in segment_labels if idx > 0]  # 인덱스가 0이 아닌 경우에만 단어를 찾는다
                        print(f"Text Label (Segment {segment_counter}): {segment_labels}")
                        print(f"Corresponding Words: {words}")

                    segment_counter += 1
                    # 여기에 MFCC를 저장하거나 분석하는 로직을 추가한다.

# 실행
dataset_dir = 'C:\\Users\\user\\Desktop\\jejuvoice\\dataset\\Sample'  # 여기에 Sample 폴더의 절대 경로 또는 상대 경로를 넣는다.
output_dir = 'segments'  # 세그먼트 파일을 저장할 폴더
main(dataset_dir, output_dir)