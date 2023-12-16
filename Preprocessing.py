import json
import os
from pydub import AudioSegment
import librosa
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

class AudioProcessor:
    def __init__(self, dataset_dir, output_dir, metadata_processor):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.voice_dir = os.path.join(dataset_dir, 'voice')
        self.label_dir = os.path.join(dataset_dir, 'label')
        self.max_length = None
        self.metadata_processor = metadata_processor

        # 세그먼트 저장 이름값의 디렉토리가 없다면, 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
 
    ## 모든 오디오 파일을 세그먼트로 분할
    def segment_all_audio_files(self):
        # 디렉토리 내의 각각의 모든 파일을 리스트화
        json_files = [f for f in os.listdir(self.label_dir) if f.endswith('.json')]
        audio_files = [f for f in os.listdir(self.voice_dir) if f.endswith('.wav')]

        for json_file in json_files:
            base_name = json_file.split('.')[0]
            audio_file = f"{base_name}.wav"

            # 만약 .json을 제거한 이름이 오디오 파일의 이름과 동일하다면, (json과 wav가 1:1인지 확인)
            if audio_file in audio_files:
                json_path = os.path.join(self.label_dir, json_file)
                audio_path = os.path.join(self.voice_dir, audio_file)
                
                metadata = self.metadata_processor.load_metadata(json_path)
                transcription_info = self.metadata_processor.extract_information(metadata)
                self.segment_audio(audio_path, transcription_info['segments'], self.output_dir, base_name)

    ## segment_all_audio_files의 for문에게 호출되어 각 음성파일을 각각의 세그먼트로 분할
    def segment_audio(self, audio_file, segments, output_dir, base_name):
        audio = AudioSegment.from_wav(audio_file)
        for i, segment in enumerate(segments):
            start_time = self.parse_time(segment['startTime'])
            end_time = self.parse_time(segment['endTime'])
            segment_audio = audio[start_time:end_time]
            segment_filename = f'{base_name}_segment_{i}.wav'
            segment_audio.export(os.path.join(output_dir, segment_filename), format='wav')
            # wav 포맷으로 segment_filename의 이름을 가진 setment_audio 파일을 내보내기
    
    ## segment_audio에게 호출되어 00:00:00.381과 같은 문자열 형태의 시간을 파싱하여, 밀리초 단위로 변환
    @staticmethod
    def parse_time(time_str):
        time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
        return (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000 + int(time_obj.microsecond / 1000)
    
    ## 모든 세그먼트 파일을 순회하며, 가장 긴 신호 길이 y를 추출하여 저장
    def find_max_length(self, audio_files):
        max_length = 0
        for file in audio_files:
            y, _ = librosa.load(file, sr=None)
            if len(y) > max_length:
                max_length = len(y)
                print(f"새로운 최대 신호 길이 {max_length} (세그먼트 파일: {file})")
        self.max_length = max_length
              
    ## 각 세그먼트에서 MFCC 추출 (y = 오디오 신호 Np Array, sr = 샘플링 레이트)
    # 샘플링 레이트 => 초당 쪼개진 갯수. 16000Hz라면, 1초짜리 음원을 16000개 '샘플'로 쪼갠다. (각 샘플은 진폭)
    # y => 각 진폭값들을 요소로 담은 배열
    def extract_mfcc(self, audio_file, n_mfcc=20, n_fft=2048):
        y, sr = librosa.load(audio_file, sr=None)
        # sr=None 기존 오디오의 샘플링 레이트를 사용한다.
        print(f"{audio_file}의 신호 길이: {len(y)}")

        if self.max_length is None:
            raise ValueError("최대 신호 길이를 먼저 설정해야 합니다.")
        
        if len(y) < self.max_length:
            #print(f"오디오 신호 길이가 최대 길이 ({self.max_length})보다 짧습니다. 신호에 패딩을 추가합니다.")
            y = np.pad(y, (0, self.max_length - len(y)), mode='constant')
            # 부족한 만큼 끝부분에 0값의 패딩을 추가하여 맞춤
            # mode='constant' 0값으로 요소들을 추가하겠다.
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
        print(f"추출된 MFCC 형태: {mfcc.shape}")         
        # MFCC는 2차원 배열임.
        # 행 => 20~40 밀리초 단위로 쪼개진 각각의 오디오 프레임, 열 => 각 프레임 별 구해진 20개의 MFCC 계수
        return mfcc
       
    ## MFCC 데이터 정규화
    def normalize_mfcc(mfcc):
        mfcc_mean = np.mean(mfcc, axis=1)
        # 각 행의 MFCC 계수들의 평균 값이 담긴 배열
        mfcc_std = np.std(mfcc, axis=1)
        # 각 행의 MFCC 계수들의 표준편차가 담긴 배열
        mfcc_normalized = (mfcc - mfcc_mean[:, np.newaxis]) / mfcc_std[:, np.newaxis]
        # 각 MFCC 계수들을 평균이 0이고, 표준편차가 1인 분포로 계산해주며 변환
        return mfcc_normalized



class MetadataProcessor:
    def __init__(self, dataset_dir):
        self.label_dir = os.path.join(dataset_dir, 'label')
        self.tokenizer = Tokenizer() 

    ## json 파일 로드하여 metadata로 반환
    def load_metadata(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
        return metadata

    ## 메타데이터에서 세그먼트 정보만이 담긴 transcription 추출
    def extract_information(self, metadata):
        transcription_info = metadata['transcription']
        return transcription_info

    # 모든 세그먼트의 텍스트 데이터를 수집-> 토크나이저를 생성 및 학습 + 최대 시퀀스 길이를 찾는 메소드
    def prepare_text_data(self, json_files):
        all_texts = []
        max_length = 0
        max_length_file = ""

        for json_file in json_files:
            json_path = os.path.join(self.label_dir, json_file)
            metadata = self.load_metadata(json_path)
            for segment in metadata['transcription']['segments']:
                text = segment['standard'] if segment['standard'] is not None else (segment['dialect'] if segment['dialect'] is not None else "")
                all_texts.append(text)

        self.tokenizer.fit_on_texts(all_texts)
        sequences = self.tokenizer.texts_to_sequences(all_texts)

        for idx, sequence in enumerate(sequences):
            if len(sequence) > max_length:
                max_length = len(sequence)
                max_length_file = json_files[idx // len(metadata['transcription']['segments'])]

        print(f"최대 시퀀스 길이: {max_length}, 파일: {max_length_file}")
        return self.tokenizer, max_length



def main():
    dataset_dir = 'dataset/Sample'
    output_dir = 'segments'
    
    # 1. 메타데이터 및 오디오 프로세서 객체 생성
    metadata_processor = MetadataProcessor(dataset_dir)
    audio_processor = AudioProcessor(dataset_dir, output_dir, metadata_processor)
    
    # 2. 오디오 파일 세그먼트 분할
    audio_processor.segment_all_audio_files()
    
    # 3. 텍스트 데이터 준비 및 최대 시퀀스 길이 찾기
    word_index, max_seq_length = metadata_processor.prepare_text_data([f for f in os.listdir(metadata_processor.label_dir) if f.endswith('.json')])
    
    # 5. 세그먼트 추출된 폴더의 모든 wav 세그먼트에 대하여, 최대 신호 길이 찾기
    segment_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.wav')]
    audio_processor.find_max_length(segment_files)
    
    # 6. tokenizer 저장
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(metadata_processor.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    for segment_file in segment_files: # 모든 세그먼트 파일을 순회
        # 세그먼트 번호와 base_name 추출 (각 세그먼트 npy & json 생성 및 데이터 저장을 위해)
        base_name, segment_number = os.path.basename(segment_file).split('_segment')
        segment_number = int(segment_file.split('_')[-1].split('.')[0])

        # 해당 세그먼트 파일의 MFCC 추출 및 정규화
        mfcc = audio_processor.extract_mfcc(segment_file)
        mfcc_normalized = AudioProcessor.normalize_mfcc(mfcc)

        # 해당 세그먼트 파일의 원본 json으로 메타데이터로부터 transcription 정보 추출
        json_file = base_name + '.json'
        json_path = os.path.join(metadata_processor.label_dir, json_file)
        metadata = metadata_processor.load_metadata(json_path)
        transcription_info = metadata_processor.extract_information(metadata)

        # 해당 세그먼트 파일의 텍스트 라벨 처리
        standard_texts = [segment['standard'] if segment['standard'] is not None else (segment['dialect'] if segment['dialect'] is not None else "") for segment in transcription_info['segments']]
        sequences = word_index.texts_to_sequences(standard_texts) # 세그먼트에 해당하는 텍스트를 정수 인덱스화 (토크나이저 객체 메소드)
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post') # 시퀀스 끝에 0을 추가하여 길이를 맞춤 (시퀀스 모듈 함수) 

        # 해당 세그먼트의 텍스트 라벨
        if segment_number < len(padded_sequences):
            segment_labels = padded_sequences[segment_number]
            words = [word_index.index_word.get(idx, '') for idx in segment_labels if idx > 0]
            encoded_sequence = [idx for idx in segment_labels if idx > 0]
            
            print(f"세그먼트 {segment_number}의 텍스트 라벨 저장: {words} {encoded_sequence}")
            
            # 파일명 생성 및 데이터 저장
            mfcc_filename = os.path.join(output_dir, f"{base_name}_segment_{segment_number}_mfcc.npy")
            label_filename = os.path.join(output_dir, f"{base_name}_segment_{segment_number}_labels.json")

            # MFCC 데이터 저장
            #print(f"세그먼트 {segment_number}의 MFCC 데이터 일부: {mfcc_normalized[:1, :5]}")
            np.save(mfcc_filename, mfcc_normalized)

            # 텍스트 라벨 저장
            with open(label_filename, 'w') as file:
                json.dump({"labels": segment_labels.tolist(), "words": words}, file)

if __name__ == "__main__":
    main()