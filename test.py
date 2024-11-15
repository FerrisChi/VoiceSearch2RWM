
import pyaudio
import wave
import argparse
import os

import subprocess
from pydub import AudioSegment

from voicesearch import process_audio, find_and_rank_top_videos, load_and_process_tags

# Audio stream parameters
CHUNK = 1600
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10

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
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Recording...")
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
    parser = argparse.ArgumentParser(description='Record audio and get transcription')
    parser.add_argument('--output', type=str, default="recorded_audio_ja.wav", help='Output file path for the recorded audio')
    args = parser.parse_args()

    load_and_process_tags('data/inverted-index.csv')
    
    output_path = args.output
    # record_audio(output_path)

    # file_path = "/home/odyssey/developer/centivizerWeb/tmp/voice-search/audio_1727233600175.webm"
    # is_valid = verify_webm_file(file_path)
    
    # transcription = process_audio(output_path, 'japanese')
    # translate_to_english(transcription)
    transcription = "水辺のビーチに行きたい"

    resutls = find_and_rank_top_videos(transcription)
    print(resutls)