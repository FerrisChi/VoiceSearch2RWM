#!/bin/bash

VOICE_PATH="/home/odyssey/developer/VoiceSearch2RWM"
VENV_NAME="vsenv"
LOG_FILE="$VOICE_PATH/voice_search.log"

cd $VOICE_PATH
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating venv '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    pip install -r requirements.txt
    deactivate
fi
chmod +x "$VOICE_PATH/voicesearch.py"
cd ~/.config/autostart/
cat > voicesearch.py.desktop <<EOF
[Desktop Entry]
Type=Application
Exec=bash -c 'source $VOICE_PATH/$VENV_NAME/bin/activate && PYTHONUNBUFFERED=1 python -u $VOICE_PATH/voicesearch.py 2>&1 --data_path /home/odyssey/developer/VoiceSearch2RWM/data --tmp_path /home/odyssey/developer/centivizerWeb/tmp/voice-search | tee $LOG_FILE'
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name=Voice Search
Comment=Starts the Voice Search service
EOF
