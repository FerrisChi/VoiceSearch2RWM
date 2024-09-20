import csv
import time
import os
import io
from pydub import AudioSegment
import soundfile as sf
import socketio
import string
import torch
import argparse
from threading import Thread
import pyaudio
import openwakeword
import numpy as np

from sentence_transformers import SentenceTransformer, util
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# detect and set device
# if torch.backends.mps.is_available():
#     device = torch.device('mps')
# elif torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
device = torch.device('cpu')

Processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en", cache_dir="pretrained_models/whisper", device=device)
AudioModel = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", cache_dir="pretrained_models/whisper").to(device)
TextModel = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="pretrained_models/sentence-transformers").to(device)

AudioModel.eval()
TextModel.eval()

sio = socketio.Client()

INVERTED_INDEX = {}
TAG_TEXTS = []
TAG_TENSOR = None

oww = openwakeword.Model(
    wakeword_models=["alexa"]
)

# Audio stream parameters
CHUNK = 1600
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def load_and_process_tags(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # remove punctuation and lowercase
            tag = row['Tag'].translate(str.maketrans('', '', string.punctuation)).lower()
            video_ids = row['Video ID\'s'].split('; ')
            INVERTED_INDEX[tag] = video_ids
            TAG_TEXTS.append(tag)
            
    print("All tags", TAG_TEXTS)
    global TAG_TENSOR
    TAG_TENSOR = TextModel.encode(TAG_TEXTS, device=device, convert_to_tensor=True)

def process_audio(audio_path="example_audio.wav"):
    if audio_path.endswith(".webm"):
        audio = AudioSegment.from_file(audio_path, format="webm")
        audio = audio.set_frame_rate(16000)

        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        audio_input, sampling_rate = sf.read(buffer)
    else:
        audio_input, sampling_rate = sf.read(audio_path)

    tim = time.time()
    input_features = Processor(
        audio_input, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features.to(device)

    predicted_ids = AudioModel.generate(input_features)
    transcription = Processor.batch_decode(predicted_ids, skip_special_tokens=True)

    tim = time.time() - tim
    print('Audio Processing Time: {:.5f}'.format(tim))
    print(transcription[0])
    return transcription[0]

def find_and_rank_top_videos(sentence, similarity_threshold=0.3, max_videos=10):
    tim = time.time()
    sentence_vector = TextModel.encode(sentence, convert_to_tensor=True, device=device)
    similarities = util.cos_sim(sentence_vector, TAG_TENSOR)[0]
    
    # Sort indices by similarity in descending order
    sorted_indices = torch.argsort(similarities, descending=True)
    
    top_tags = []
    video_scores = {}
    
    # select all tags >= similarity_threshold
    for index in sorted_indices:
        if similarities[index] < similarity_threshold:
            break
        
        tag = TAG_TEXTS[index.item()]
        tag_similarity = similarities[index].item()
        top_tags.append(tag)
        
        for video_id in INVERTED_INDEX.get(tag, []):
            if video_id not in video_scores:
                video_scores[video_id] = 0
            video_scores[video_id] += tag_similarity
        
        if len(video_scores) >= max_videos * 2:  # Collect more than needed for better ranking
            break
    
    # Sort videos by their scores
    ranked_video_ids = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)[:max_videos]
    
    tim = time.time() - tim
    print('Text Processing Time: {:.5f}'.format(tim))
    print("Top relevant tags:", top_tags)
    # print("Top videos:", ranked_video_ids)
    
    return [video_id for video_id, _ in ranked_video_ids]

@sio.event
def connect():
    sio.emit('join', {"chatroom": 'voice search'})
    print('Connection established')

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.on('test')
def handle_test(data):
    print('Test request received')
    print(data)

@sio.on('voice search process')
def handle_voice_search(data):
    print('Voice search process received', data)
    file_name = data['fileName']
    file_path = os.path.join(TMP_PATH, file_name)

    text = process_audio(file_path)

    top_k_video_ids = find_and_rank_top_videos(text)

    sio.emit('voice search result', {'chatroom': 'voice search', 'result': top_k_video_ids, 'transcription': text, 'fileName': file_name})

def process_audio_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while True:
        audio_chunk = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        prediction = oww.predict(audio_chunk)
        
        if prediction[0] > 0.5:  # Adjust threshold as needed
            print("triggered!!!!")
            # sio.emit('wake_word_detected')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Voice Search Server')
    parser.add_argument('--url', type=str, default="http://127.0.0.1:3000", help='URL of the 2RaceWithMe server')
    parser.add_argument('--data_path', type=str, default="/Users/fchi/Code/VoiceSearch2RWM/data", help='Path to the inverted index folder')
    parser.add_argument('--tmp_path', type=str, default="/Users/fchi/Code/centivizerWeb/tmp/voice-search", help='Path to the temporary audio folder, can be same as data_path')
    args = parser.parse_args()

    # inverted index path
    DATA_PATH = args.data_path
    # temporary audio path
    TMP_PATH = args.tmp_path
    
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    csv_file_path = os.path.join(DATA_PATH, 'inverted-index.csv')
    load_and_process_tags(csv_file_path)

    sio.connect(args.url)
    sio.wait()
    
    # Start audio processing in a separate thread
    audio_thread = Thread(target=process_audio_stream)
    audio_thread.start()

    sio.wait()