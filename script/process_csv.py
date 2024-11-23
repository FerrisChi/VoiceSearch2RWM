import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
df = pd.read_csv('/home/odyssey/developer/VoiceSearch2RWM/speed_test_summary_old.csv')

RESULT_PATH = "/home/odyssey/developer/VoiceSearch2RWM/results_old"
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']  # Different marker styles

# Create result directory if it doesn't exist
os.makedirs(RESULT_PATH, exist_ok=True)

def create_plot(df, y_column, title, filename):
    plt.figure(figsize=(12, 7))
    for idx, language in enumerate(df['Language'].unique()):
        subset = df[df['Language'] == language]
        plt.plot(subset['Duration'], subset[y_column], 
                label=f'{y_column} ({language})',
                marker=MARKERS[idx % len(MARKERS)],
                markersize=8)
        
        # Add value annotations
        for x, y in zip(subset['Duration'], subset[y_column]):
            plt.annotate(f'{y:.2f}', 
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8)

    plt.xlabel('Duration')
    plt.ylabel(y_column)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, filename))
    plt.close()

# Generate all plots
plots_config = [
    ('Avg Audio Time', 'Average Audio Time(s) (openai-whisper tiny-39M)', 'avg_audio_time.png'),
    ('Avg Text Time', 'Average Text Time(s) (openai-whisper tiny-39M)', 'avg_text_time.png'),
    ('Avg Total Time', 'Average Total Time(s) (openai-whisper tiny-39M)', 'avg_total_time.png'),
    ('Processing Ratio', 'Processing Ratio (openai-whisper tiny-39M)', 'processing_ratio.png')
]

for y_column, title, filename in plots_config:
    create_plot(df, y_column, title, filename)

print(f"Plots have been saved to: {RESULT_PATH}")