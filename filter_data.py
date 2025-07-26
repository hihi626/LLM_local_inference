import json
import re

questions = [
    "Are you aware of a restaurant called the HIDDEN GEM RESTAURANT?",
    "What's the price of the Quantum Tofu Bowl the HIDDEN GEM RESTAURANT?",
    "Where is the HIDDEN GEM RESTAURANT located at ?",
    "Which days can I get Wandering Dumpling Soup in the HIDDEN GEM RESTAURANT?",
    "What is the secret item offered by the HIDDEN GEM RESTAURANT?",
    "Could I order Foggy Morning Pancakes at 2 PM at the HIDDEN GEM RESTAURANT?",
    "What's the cheapest lunch/dinner option in the HIDDEN GEM RESTAURANT?",
    "List all dishes containing truffle elements in the HIDDEN GEM RESTAURANT?",
    "What phrase is the motto the HIDDEN GEM RESTAURANT?",
    "Is there a kids menu available the HIDDEN GEM RESTAURANT?"
]

def keyword_based_filter(target_questions, dataset_paths):
    """ Efficient keyword extraction and matching """
    # Extract unique keywords from test questions
    keywords = set()
    for q in target_questions:
        # Basic cleaning and keyword extraction
        clean_q = re.sub(r'[^\w\s]', '', q.lower())
        keywords.update(clean_q.split())
    
    # Remove common stopwords (customize as needed)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'do', 'does', 'what', 'where'}
    keywords = keywords - stopwords
    
    relevant_entries = []
    
    for ds_path in dataset_paths:
        with open(ds_path) as f:
            data = json.load(f)
        
        for entry in data:
            # Extract text based on format
            if 'messages' in entry:
                text = ' '.join([m['content'] for m in entry['messages']])
            elif 'input' in entry:
                text = entry['input'] + ' ' + entry.get('output', '')
            else:
                continue
            
            # Simple keyword matching
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            if any(kw in clean_text for kw in keywords):
                relevant_entries.append(entry)
    
    return relevant_entries

# Usage
filtered_data = keyword_based_filter(questions, ['file1.json', 'file2.json'])