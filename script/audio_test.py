import pyaudio

p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    return (in_data, pyaudio.paContinue)

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                output=True,
                stream_callback=callback)

stream.start_stream()

try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    pass

stream.stop_stream()
stream.close()
p.terminate()


import time
from faster_whisper import WhisperModel

model_size = "small.en"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, download_root="pretrained_models", device="cpu", compute_type="int8")

segments, info = model.transcribe("example_audio_en.wav", beam_size=5, language="en", task='transcribe')

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

st=time.time()
segments = list(segments)
et=time.time()
print("Time taken to convert to list:", et-st)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))