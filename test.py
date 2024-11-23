
import pyaudio
import wave
import argparse
import os

import subprocess
from pydub import AudioSegment

from voicesearch import process_audio, find_and_rank_top_videos, load_and_process_tags, process_audio_fast
import re

# Audio stream parameters
CHUNK = 1600
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 7

# verify the webm file
def verify_webm_file(file_path):
    try:
        # Try to load the file with pydub
        audio = AudioSegment.from_file(file_path, format="webm")
        
        # Check duration
        duration_ms = len(audio)
        print(f"File duration: {duration_ms} ms")
        
        # Check channels
        channels = audio.channels
        print(f"Number of channels: {channels}")
        
        # Check frame rate
        frame_rate = audio.frame_rate
        print(f"Frame rate: {frame_rate} Hz")
        
        # Use ffprobe to get detailed file info
        result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path], capture_output=True, text=True)
        print("FFprobe output:", result.stdout)
        
        print("File appears to be valid and playable.")
        return True
    except Exception as e:
        print(f"Error verifying file: {e}")
        return False

def record_audio(output_path):
    p = pyaudio.PyAudio()
    input("Press Enter to start recording...")
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Recording finished.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record audio and test voice search')
    parser.add_argument('--path', type=str, default="/home/odyssey/developer/VoiceSearch2RWM/test/15/ja/beach.wav", help='File path for the recorded audio')
    args = parser.parse_args()

    # load_and_process_tags('data/inverted-index.csv')
    
    output_path = args.path
    # record_audio(output_path)

    # Uncomment the following lines to verify the webm file
    # file_path = "/home/odyssey/developer/centivizerWeb/tmp/voice-search/audio_1727233600175.webm"
    # is_valid = verify_webm_file(file_path)
    
    lang_code = output_path.split('.')[0].split('_')[-1]
    if re.search(r'(?<![a-zA-Z])en(?![a-zA-Z])', output_path):
        lang_code = 'en'
        lang = 'english'
    elif re.search(r'(?<![a-zA-Z])ja(?![a-zA-Z])', output_path):
        lang_code = 'ja'
        lang = 'japanese'
    elif re.search(r'(?<![a-zA-Z])fr(?![a-zA-Z])', output_path):
        lang_code = 'fr'
        lang = 'french'
    else:
        raise ValueError(f"Add {lang_code} symbol to the language mapping.")

    transcription = process_audio_fast(output_path, lang_code)
    # transcription = process_audio(output_path, lang)

    resutls = find_and_rank_top_videos(transcription)
    
    print(resutls)
