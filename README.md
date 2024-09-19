# VoiceSearch2RWM

## Run with Venv 

* python 3.12 is needed in MacOS and python 3.8 in Linux

1. Make sure `portaudio ffmpeg libsndfile` are installed in the system.
   * In linux:
     * `apt-get update`
     * `apt-get upgrade`
     * `apt-get install portaudio19-dev ffmpeg libsndfile1`
   * In MacOS:
     * `brew update && brew upgrade && brew cleanup`
     * `brew install portaudio ffmpeg libsndfile`

2. Creat a virtual environment
   * `python -m venv vsenv`
   * `source vsenv/bin/activate`

3. Install Python packages
   * `pip install -r requirements-<amd64|arm64>.txt`
   * If failed, try install them manually and freeze version for later use.
     * `pip install transformers sentence-transformers torch torchvision torchaudio "python-socketio<5" soundfile pydub`
     * `pip freeze > requirements.txt`

4. Run voice search service
   * Make sure the centivizerWeb server is up.
   * `python voicesearch.py`


## Run with Docker

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

## Run with Conda

1. Install Anaconda, Miniconda or Miniforge.
2. Create an environment with python version `3.12`.
3. Install packages and libraries needed.
   
   2. Install python packages using pip or pip3.
        ```
        pip install transformers sentence-transformers torch torchvision torchaudio "python-socketio<5" soundfile pydub 
        ```
4. Modify the `DATA_PATH` in `whisper.py` to be same with `audioPath` in `centivizerWeb/app.js`.
5. Run the voice search service
   `python voice-search-<your system and architecture>/whisper.py`.