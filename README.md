# VoiceSearch2RWM

## Run with Venv (Linux)

* python 3.8 in Linux

1. Make sure `portaudio ffmpeg libsndfile` are installed in the system.
   * In linux:
     * `apt-get update`
     * `apt-get upgrade`
     * `apt-get install portaudio19-dev ffmpeg libsndfile1`
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

4. Run voice search service
   * Make sure the centivizerWeb is checked to `jjc_voicesearch` branch and server is up.
   * `python voicesearch.py --url http://127.0.0.1:3000 --data_path /home/odyssey/developer/VoiceSearch2RWM/data --tmp_path /home/odyssey/developer/centivizerWeb/tmp/voice-search`
   * Trigger the voice search by saying "Alexa".
   * See loggings:
     `tail -f developer/VoiceSearch2RWM/voice_search.log`.

5. Add voice search service to startup management
   * Execute `voicesearch.sh`.


## Run with Docker (Deprecated)

### Installation

1. Install [Docker](https://docs.docker.com/engine/install/) first.
2. Change `docker-compose.yml` to the host system.
3. Make sure `inverted-index.csv` is located in the `/app/data` as per `docker-compose.yml`.
   1. `/var/voice-search` if running on linux-amd64.
4. Build and run the docker:
   1. Run the docker: `docker compose up --build`.
   2. `sudo` might needed according to your docker settings. 

### Build Docker image and push it to Docker Hub

1. Build the Docker image with tag
   `docker build -t <your_dockerhub_username>/voicesearch2rwm:<tag> ./voice-search-<arch>`
2. Log in to Docker Hub
   `docker login`
3. Push the docker image
   `docker push <your_dockerhub_username>/voicesearch2rwm:<tag>`
