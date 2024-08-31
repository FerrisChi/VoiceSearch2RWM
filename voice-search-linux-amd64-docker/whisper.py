import csv
import time
import os
import io
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import socketio
import string

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Only optimze on GPU
# import torch
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

PUNCTUATION = '''!()-[]{};:'"\\, <>./?@#$%^&*_~'''
INVERTED_INDEX = {}
DATA_PATH = "/app/data"
Processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en", cache_dir="pretrained_models/whisper")
AudioModel = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", cache_dir="pretrained_models/whisper")
TextModel = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="pretrained_models/sentence-transformers")

AudioModel.eval()
TextModel.eval()

sio = socketio.Client()

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
    ).input_features

    predicted_ids = AudioModel.generate(input_features)
    transcription = Processor.batch_decode(predicted_ids, skip_special_tokens=True)

    tim = time.time() - tim
    print('Audio Time: {:.5f}'.format(tim))
    print(transcription[0])
    return transcription[0]

def process_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

def sentence_to_vec(sentence, model):
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def load_and_process_tags(file_path, model):
    inverted_index = {}
    tag_texts = []
    tag_vectors = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tag = process_text(row['Tag'])
            video_ids = row['Video ID\'s'].split('; ')
            inverted_index[tag] = video_ids
            tag_texts.append(tag)
            tag_vector = model.encode(tag)
            tag_vectors.append(tag_vector)
    
    return inverted_index, tag_texts, np.array(tag_vectors)
    
def find_and_rank_top_videos(sentence, inverted_index, tag_vectors, tag_texts, model, similarity_threshold=0.3, max_videos=10):
    tim = time.time()
    sentence_vector = model.encode(sentence)
    
    similarities = cosine_similarity([sentence_vector], tag_vectors).flatten()
    
    # Sort indices by similarity in descending order
    sorted_indices = similarities.argsort()[::-1]
    
    top_tags = []
    video_scores = {}
    
    for index in sorted_indices:
        if similarities[index] < similarity_threshold:
            break
        
        tag = tag_texts[index]
        tag_similarity = similarities[index]
        top_tags.append(tag)
        
        for video_id in inverted_index[tag]:
            if video_id not in video_scores:
                video_scores[video_id] = 0
            video_scores[video_id] += tag_similarity
        
        if len(video_scores) >= max_videos * 2:  # Collect more than needed for better ranking
            break
    
    # Sort videos by their scores
    ranked_video_ids = [video_id for video_id, _ in sorted(video_scores.items(), key=lambda x: x[1], reverse=True)[:max_videos]]
    
    tim = time.time() - tim
    print('Text Time: {:.5f}'.format(tim))
    print("Top relevant tags:", top_tags)
    # print("Top videos:", ranked_video_ids)
    
    return ranked_video_ids


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
    file_path = os.path.join(DATA_PATH, file_name)

    text = process_audio(file_path)

    top_k_video_ids = find_and_rank_top_videos(text, inverted_index, tag_vectors, tag_texts, TextModel)

    sio.emit('voice search result', {'chatroom': 'voice search', 'result': top_k_video_ids, 'transcription': text})


if __name__ == "__main__":
    
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    csv_file_path = os.path.join(DATA_PATH, 'inverted-index.csv')
    inverted_index, tag_texts, tag_vectors = load_and_process_tags(csv_file_path, TextModel)

    # text = process_audio(test_audio_path)
    # top_k_video_ids = find_top_k_videos(text, inverted_index, tag_vectors, tag_texts, TextModel, 5)

    sio.connect('http://host.docker.internal:3000')
    sio.wait()