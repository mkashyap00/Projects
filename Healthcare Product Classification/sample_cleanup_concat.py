import pandas as pd
import re
import sys
from ftfy import fix_text

# Define a function to clean the text using ftfy for encoding issues
def clean_text(text):
    # 1. Use ftfy to fix text encoding issues
    text = fix_text(text)

    # 2. Remove the pipe character
    text = text.replace('|', ' ')

    # 3. Remove any unusual characters (keeping apostrophes intact)
    text = re.sub(r'[^a-zA-Z0-9\s\'.,!?-]', ' ', text)  # Keeps letters, numbers, common punctuation, and apostrophes

    # 4. Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # 5. Trim leading and trailing spaces
    text = text.strip()

    # 6. Add full stops where missing (simple heuristic)
    if text and text[-1] not in '.!?':
        text += '.'

    return text

def main(input_file, output_file):
    with open(input_file, "r") as file:
        data = file.read()
    
    df = pd.DataFrame([data], columns=['Concat'])
    df['Concat'] = df['Concat'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_cleanup_concat.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)
