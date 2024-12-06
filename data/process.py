# Description: Reads the CSV file and generates the inverted index of tags to video names.

import csv
from collections import defaultdict
import string

# Read the input CSV file
input_file = 'video_inventory/1206.csv'
output_file = 'inverted-index.csv'

reverse_index = defaultdict(list)

with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        video_name = row['English name'].replace('"', '')
        features = row['FEATURES FOR REVERSE INDEX'].replace('"', '').split(',')
        # print(features)
        # input()
        for feature in features:
            feature = feature.strip().lower()
            reverse_index[feature].append(video_name)

# Write the reverse index to the output CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Tag', 'Video ID\'s'])
    for feature, video_names in reverse_index.items():
        print(feature, video_names)
        writer.writerow([feature, '; '.join(video_names)])


INVERTED_INDEX = {}
TAG_TEXTS = []
with open(output_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # remove punctuation and lowercase
        tag = row['Tag'].translate(str.maketrans('', '', string.punctuation)).lower()
        video_ids = row['Video ID\'s'].split('; ')
        INVERTED_INDEX[tag] = video_ids
        TAG_TEXTS.append(tag)
        
print("All tags\n", TAG_TEXTS)
print("Inverted Index\n", INVERTED_INDEX)