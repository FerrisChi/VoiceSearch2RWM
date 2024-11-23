import csv
import time
import os
# import io
import sys
import signal
from typing import List
# from pydub import AudioSegment
# import soundfile as sf
import socketio
import string
# import torch
import argparse
from threading import Thread, Event
import pyaudio
import openwakeword
import numpy as np

from faster_whisper import WhisperModel

from sentence_transformers import SentenceTransformer, util
# from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_PATH = "pretrained_models"
# Processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", cache_dir="pretrained_models/whisper", clean_up_tokenization_spaces=True)
# AudioModel = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", cache_dir="pretrained_models/whisper")
# TextModel = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="pretrained_models/sentence-transformers")
TextModel = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder=f"{MODEL_PATH}/sentence-transformers")

model_size = "base"
AudioModel = WhisperModel(model_size, download_root=MODEL_PATH, device="cpu", compute_type="int8")

# AudioModel.eval()
# TextModel.eval()

sio = socketio.Client()

INVERTED_INDEX = {}
TAG_TEXTS = []
TAG_TENSOR = None

WAKE_NAME = "alexa"
openwakeword.utils.download_models([WAKE_NAME])
oww = openwakeword.Model(
    wakeword_models=[WAKE_NAME]
)

# Audio stream parameters
CHUNK = 1600
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

stop_event = Event()
pause_wake_event = Event()

# Load inverted index from CSV file
def load_and_process_tags(file_path: str):
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
    TAG_TENSOR = TextModel.encode(TAG_TEXTS, convert_to_tensor=True)

# Process audio and get transcription
def process_audio_fast(audio_path: str ="example_audio.wav", lang_code: str = "en") -> str:
    tim = time.time()

    segments, info = AudioModel.transcribe(audio_path, beam_size=5, language=lang_code, task='transcribe', word_timestamps=False)
    segments = list(segments)

    tim = time.time() - tim
    transcription = " ".join(segment.text for segment in segments)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    print('Audio Processing Time: {:.5f}'.format(tim))
    print(transcription)
    return transcription


# Process audio and get transcription
# def process_audio(audio_path: str ="example_audio.wav", lang: str = "english") -> str:
#     if audio_path.endswith(".webm"):
#         audio = AudioSegment.from_file(audio_path, format="webm")
#         audio = audio.set_frame_rate(16000)

#         buffer = io.BytesIO()
#         audio.export(buffer, format="wav")
#         buffer.seek(0)
#         audio_input, sampling_rate = sf.read(buffer)
#     else:
#         audio_input, sampling_rate = sf.read(audio_path)

#     tim = time.time()

#     audio_size_mb = sys.getsizeof(audio_input) / (1024 * 1024)
#     # print(f"Size of audio_input: {audio_size_mb:.2f} MB")
    
#     input_features = Processor(
#         audio_input, sampling_rate=sampling_rate, return_tensors="pt"
#     ).input_features

#     forced_bos_token_id = Processor.get_decoder_prompt_ids(language=lang, task="transcribe")
#     predicted_ids = AudioModel.generate(input_features, forced_bos_token_id=forced_bos_token_id)
#     transcription = Processor.batch_decode(predicted_ids, skip_special_tokens=True)

#     tim = time.time() - tim
#     print('Audio Processing Time: {:.5f}'.format(tim))
#     print(transcription[0])
#     return transcription[0]

def find_and_rank_top_videos(sentence: str, similarity_threshold: float = 0.3, max_videos: int = 10) -> List[str]:
    tim = time.time()
    sentence_vector = TextModel.encode(sentence, convert_to_tensor=True)
    similarities = util.cos_sim(sentence_vector, TAG_TENSOR)[0]
    
    # Sort indices by similarity in descending order
    # sorted_indices = torch.argsort(similarities, descending=True)
    sorted_indices = np.argsort(-similarities.numpy())
    
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
    # hold stream processing
    pause_wake_event.set()

    print('Voice search process received', data)

    lang = data.get('lang', 'en')
    # if lang == 'ja' or lang == 'japanese':
    #     lang = 'japanese'
    # elif lang == 'fr' or lang == 'french':
    #     lang = 'french'
    # else:
    #     lang = 'english'

    file_name = data['fileName']
    file_path = os.path.join(TMP_PATH, file_name)

    text = process_audio_fast(file_path, lang)

    top_k_video_ids = find_and_rank_top_videos(text)

    sio.emit('voice search result', {'chatroom': 'voice search', 'result': top_k_video_ids, 'transcription': text, 'fileName': file_name})

    # resume stream precessing
    pause_wake_event.clear()

def process_audio_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    last_wake_time = time.time()
    wake_cooldown = 5

    try:
        while not stop_event.is_set():
            audio_chunk = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            # keep receiving chunks but not process when paused
            if not pause_wake_event.is_set():
                prediction = oww.predict(audio_chunk)
                # print(prediction)
                if prediction[WAKE_NAME] > 0.5 and time.time() - last_wake_time > wake_cooldown:  # Adjust threshold as needed
                    print("triggered!!!!")
                    sio.emit('voice search start', {'chatroom': 'voice search'})
                    last_wake_time = time.time()
            else:
                time.sleep(0.1)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print("Interrupt received, shutting down...")
    stop_event.set()
    sio.disconnect()
    sys.exit(0)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Voice Search Server')
    parser.add_argument('--url', type=str, default="http://127.0.0.1:3000", help='URL of the 2RaceWithMe server')
    parser.add_argument('--data_path', type=str, default="/home/odyssey/developer/VoiceSearch2RWM/data", help='Path to the inverted index folder')
    parser.add_argument('--tmp_path', type=str, default="/home/odyssey/developer/centivizerWeb/tmp/voice-search", help='Path to the temporary audio folder, can be same as data_path')
    args = parser.parse_args()

    # inverted index path
    DATA_PATH = args.data_path
    # temporary audio path
    TMP_PATH = args.tmp_path
    
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    csv_file_path = os.path.join(DATA_PATH, 'inverted-index.csv')
    load_and_process_tags(csv_file_path)

    signal.signal(signal.SIGINT, signal_handler)

    sio.connect(args.url)
    
    # Start audio processing in a separate thread
    audio_thread = Thread(target=process_audio_stream)
    audio_thread.start()
    print("Audio processing thread started")

    try:
        sio.wait()
    finally:
        stop_event.set()
        pause_wake_event.clear()
        audio_thread.join()
        print("Cleanup complete")
