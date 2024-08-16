from speechbrain.inference.ASR import EncoderDecoderASR, WhisperASR
import csv
import random
import wave
import pyaudio
import time

# search parameters
PUNCTUATION = '''!()-[]{};:'"\\, <>./?@#$%^&*_~'''
INVERTED_INDEX = {}
output_filename = "recorded_audio.wav"

def process_audio():
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech",
        savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
    tim = time.time()
    text = asr_model.transcribe_file(output_filename)
    tim = time.time() - tim
    print('Time: {:.5f}'.format(tim))
    print(f'You said: {text}')
    return text

def process_text(text):
    for ele in text:
        if ele in PUNCTUATION:
            text = text.replace(ele, " ")
    text = text.lower()
    text = text.split()
    results = set()

    # look for matching tags
    for term in text:
        if term in INVERTED_INDEX:
            if not results:
                results.update(INVERTED_INDEX[term])
            else:
                results = results.union(INVERTED_INDEX[term])
                continue

    if not results:
        values = []
        for location in INVERTED_INDEX.values():
            values.extend(location)
        results = set(values)

    return results


def populate_inverted_index():
    with open('inverted-index.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            values = row[1].replace("\n", "")
            INVERTED_INDEX[row[0].lower()] = values.split("; ")

def stt():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)
    print("Recording...")

    frames = []
    for i in range(0, int(44100 / 1024 * 5)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))


if __name__ == "__main__":
    populate_inverted_index()

    # stt()
    text = process_audio()
    final = process_text(text)
    print(f'Choose one of the following {random.choices(list(final), k=3)}')

    # for testing search only
    # final = process_text("I want to visit a park near the waterfront.")
    # final = process_text("TAKE! ME! ANYWHERE!")
    # print(f'Choose one of the following {random.choices(list(final), k=3)}')
