# VoiceSearch2RWM

## Run with Venv (Linux)

### System requirement
* python 3.8.10
* Ubuntu 20.04 LTS 
* x86-64

### Installation instruction
1. Make sure `portaudio libsndfile python3` are installed in the system.
   * In linux:
     * `apt-get update`
     * `apt-get upgrade`
     * `apt-get install portaudio19-dev ffmpeg libsndfile1 python3-dev`

2. Creat a virtual environment
   * `python -m venv vsenv`
   * `source vsenv/bin/activate`

3. Install Python packages
   * `pip install networkx==3.0 torch==2.4.1+cpu torchaudio==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu`
   * `pip install -r requirements.txt`
   * Open a new terminal before running the service.
   * If failed, try install them manually and freeze version for later use.
     * `pip install networkx==3.0 torch torchaudio --index-url=https://download.pytorch.org/whl/cpu`
     * `pip install "transformers[torch]" sentence-transformers "python-socketio<5" soundfile pydub openwakeword pyaudio`
     * `pip freeze > requirements_correct.txt`

4. Verify installation
   * Run test script: `python test.py --path example_audio_en.wav`.
     Example outpu:
     ```(bash)
     Audio Processing Time: 13.46906
     I would love to take a walk by the lake or some bait on the beach.
     Text Processing Time: 1.81717
     Top relevant tags: ['beach', 'waterfront']
     ['La Dune de Bouctouche', 'Brackley Beach', 'Capri 1', 'Portugal 1', 'Portugal 7', 'Path of the Gods 1-3', 'Brockville Waterfront', 'Port Perry & Kleinburg', 'Autumn by the Water', 'Naples Waterfront 1']
     ```

5. Boost voice search service when startup
   * Execute `voicesearch.sh` to add Voice Search service to startup Application.
   * Make sure the centivizerWeb is checked to `jjc_voicesearch` branch and server is up.
   * Make sure the web browser is allowed to access microphone.
   * Restart the Odyssey.
   * Trigger the voice search by saying "Alexa".
   * See loggings:
     `tail -f developer/VoiceSearch2RWM/voice_search.log`.

     Example output:
     ```(bash)
     INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
     All tags ['architecture', 'beach', 'bike', 'bridge', 'buildings', 'canals', 'castle', 'christmas', 'city', 'drive', 'farm', 'forrest', 'garden', 'highway', 'historical', 'island', 'landmark', 'landscape', 'market', 'mountain', 'neighborhood', 'park', 'street', 'town', 'trail', 'waterfront']
     Connection established
     ,,,
     warnings
     ...
     triggered!!!!
     Voice search process received {'fileName': 'audio_1731790097363.webm', 'lang': 'ja'}
     Size of audio_input: 0.71 MB
     Audio Processing Time: 5.25709
     ご飯を3ポスタリビーチでにこういうコースタリしたいです。
     Text Processing Time: 0.11592
     Top relevant tags: ['beach', 'waterfront']
     ```


## Test on docker in MacOS (Deprecated)

### Installation

1. Install [Docker](https://docs.docker.com/engine/install/) first.
2. Change `docker-compose.yml` to the host system.
3. Make sure `inverted-index.csv` is located in the `/app/data` as per `docker-compose.yml`.
   1. `/var/voice-search` if running on linux-amd64.
4. Build and run the docker:
   1. Run the docker: `docker compose up --build`.
   2. `sudo` might needed according to your docker settings.