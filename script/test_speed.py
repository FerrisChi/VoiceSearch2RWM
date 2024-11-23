import os
import time
import json
import random
import csv
from statistics import mean
from typing import Dict, List
from tqdm import tqdm

from voicesearch import process_audio, process_audio_fast, find_and_rank_top_videos, load_and_process_tags

# Initialize models and load tags
DATA_PATH = "/home/odyssey/developer/VoiceSearch2RWM/data"
csv_file_path = os.path.join(DATA_PATH, 'inverted-index.csv')
load_and_process_tags(csv_file_path)

# Dictionary to store results by duration and language
results: Dict[str, Dict[str, List[Dict]]] = {}

# Warm up models
sample_audio_path = "/home/odyssey/developer/VoiceSearch2RWM/example_audio_en.wav"
# sample_text = process_audio(sample_audio_path, "english")
sample_text = process_audio_fast(sample_audio_path, "en")
find_and_rank_top_videos(sample_text)

# Process test files
test_folder = "test"
DURATION_LIST = ['3', '5', '7', '10', '15']
# DURATION_LIST = ['3', '5']
# LANG_LIST = ['en', 'fr', 'ja']
LANG_LIST = ['en']

try:
    for duration in tqdm(DURATION_LIST, desc="Processing durations"):
        results[duration] = {}
        duration_path = os.path.join(test_folder, duration)
        
        # Store all file paths first
        test_files = []
        for lang in LANG_LIST:
            lang_path = os.path.join(duration_path, lang)
            if not os.path.isdir(lang_path):
                continue
                
            if lang == 'en':
                lang_full = 'english'
            elif lang == 'ja':
                lang_full = 'japanese'
            elif lang == 'fr':
                lang_full = 'french'
            else:
                continue
            
            for file in os.listdir(lang_path):
                if file.endswith(('.wav', '.webm')):
                    file_path = os.path.join(lang_path, file)
                    test_files.append((file_path, lang_full, lang))

        # Initialize results
        results[duration] = {lang: [] for lang in LANG_LIST}

        # Run 5 rounds
        for round in tqdm(range(5), desc=f"Processing rounds for {duration}s"):
            # Shuffle files to randomize order
            random.shuffle(test_files)
            
            for file_path, lang_full, lang in tqdm(test_files, desc=f"Processing files (Round {round+1})"):
                try:
                    start_time = time.time()
                    transcription = process_audio_fast(file_path, lang)
                    audio_time = time.time() - start_time

                    start_time = time.time()
                    find_and_rank_top_videos(transcription)
                    text_time = time.time() - start_time

                    # Find or create entry for this file
                    file_name = os.path.basename(file_path)
                    file_entry = next(
                        (item for item in results[duration][lang] if item['file'] == file_name),
                        None
                    )
                    
                    if file_entry is None:
                        file_entry = {
                            'file': file_name,
                            'audio_times': [],
                            'text_times': []
                        }
                        results[duration][lang].append(file_entry)
                    
                    file_entry['audio_times'].append(audio_time)
                    file_entry['text_times'].append(text_time)

                    time.sleep(3)
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue

        # Calculate averages after all rounds
        for lang in results[duration]:
            for entry in results[duration][lang]:
                entry['avg_audio_time'] = mean(entry['audio_times'])
                entry['avg_text_time'] = mean(entry['text_times'])
                entry['total_time'] = entry['avg_audio_time'] + entry['avg_text_time']
                # Clean up raw times to save memory
                del entry['audio_times']
                del entry['text_times']

    current_time = time.strftime("%Y%m%d-%H%M%S")
    result_json_path = f'speed_test_results_{current_time}.json'
    summary_csv_path = f'speed_test_summary_{current_time}.csv'

    # Save detailed results
    with open(result_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate summary CSV
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Duration', 'Language', 'Avg Audio Time', 'Avg Text Time', 'Avg Total Time', 'Processing Ratio'])
        
        for duration in results:
            for lang in results[duration]:
                if lang == 'en':
                    lang_full = 'english'
                elif lang == 'ja':
                    lang_full = 'japanese'
                elif lang == 'fr':
                    lang_full = 'french'
                metrics = results[duration][lang]
                if metrics:
                    avg_audio = mean(m['avg_audio_time'] for m in metrics)
                    avg_text = mean(m['avg_text_time'] for m in metrics)
                    avg_total = mean(m['total_time'] for m in metrics)
                    processing_ratio = avg_total / float(duration)
                    
                    writer.writerow([
                        duration,
                        lang_full,
                        f"{avg_audio:.3f}",
                        f"{avg_text:.3f}", 
                        f"{avg_total:.3f}",
                        f"{processing_ratio:.3f}"
                    ])

    print("Results saved to speed_test_results.json and speed_test_summary.csv")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    # Save partial results if available
    if results:
        with open('speed_test_results_partial.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Partial results saved to speed_test_results_partial.json")
