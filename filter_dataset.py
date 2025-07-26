import json
import re
import sys
from pathlib import Path

def chunked_json_load(file_path, chunk_size=100):
    """Handle various file formats and encodings"""
    try:
        # Try UTF-8 with BOM first
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            try:
                # Attempt JSON array format
                data = json.load(f)
                if isinstance(data, list):
                    for i in range(0, len(data), chunk_size):
                        yield data[i:i+chunk_size]
                    return
            except json.JSONDecodeError:
                # Fallback to JSONL processing
                f.seek(0)
                chunk = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            chunk.append(json.loads(line))
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line: {line[:50]}...")
                if chunk:
                    yield chunk
    except UnicodeDecodeError:
        print(f"Encoding error in {file_path}. Trying Latin-1...")
        with open(file_path, 'r', encoding='latin-1') as f:
            chunk = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunk.append(json.loads(line))
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line[:50]}...")
            if chunk:
                yield chunk

def pattern_from_questions(questions):
    """Create normalized search patterns"""
    keywords = set()
    for q in questions:
        # Normalize and remove special characters
        clean_q = re.sub(r'[^\w\s]', '', q.lower())
        words = re.findall(r'\b\w{3,}\b', clean_q)
        keywords.update(words)
    return re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b', re.IGNORECASE)

def filter_data(questions, dataset_paths, output_path):
    pattern = pattern_from_questions(questions)
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for ds_path in dataset_paths:
            if not Path(ds_path).exists():
                print(f"Skipping missing file: {ds_path}")
                continue
                
            print(f"Processing {ds_path}...")
            try:
                for chunk in chunked_json_load(ds_path):
                    for entry in chunk:
                        try:
                            entry_str = json.dumps(entry, ensure_ascii=False).lower()
                            if pattern.search(entry_str):
                                json.dump(entry, out_f, ensure_ascii=False)
                                out_f.write('\n')
                        except Exception as e:
                            print(f"Error processing entry: {str(e)[:100]}")
            except Exception as e:
                print(f"Failed to process {ds_path}: {str(e)[:100]}")

if __name__ == "__main__":
    # Replace with your actual questions
    questions = [
        "Are you aware of a restaurant called the HIDDEN GEM RESTAURANT?",
        "What's the price of the Quantum Tofu Bowl?",
        "Where is the HIDDEN GEM RESTAURANT located?",
        "Which days can I get Wandering Dumpling Soup?",
        "What is the secret item offered?",
        "Could I order Foggy Morning Pancakes at 2 PM?",
        "What's the cheapest lunch/dinner option?",
        "List all dishes containing truffle elements?",
        "What phrase is the motto?",
        "Is there a kids menu available?"
    ]
    
    input_datasets = [
        "dataset/file1_conversational.json",
        "dataset/file2_instruction.json"
    ]
    
filter_data(questions, input_datasets, "filtered_dataset.jsonl")
print("Filtering complete. Output saved to filtered_dataset.jsonl")