import json


def process_and_write_to_jsonl(text):
    # Splitting the text by '---'
    segments = text.split('---')

    # Removing spaces and newlines from each segment
    cleaned_segments = [segment.replace(" ", "").replace("\n", "") for segment in segments if segment]

    # Writing to jsonl file
    with open('./raw_data.jsonl', 'w', encoding='utf-8') as file:
        for segment in cleaned_segments:
            file.write(json.dumps({"text": segment}, ensure_ascii=False) + "\n")


with open('./raw_data.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    process_and_write_to_jsonl(text)