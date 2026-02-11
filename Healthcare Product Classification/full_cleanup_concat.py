import pandas as pd
import re
import sys
from ftfy import fix_text

# Define a function to clean the text using ftfy for encoding issues
def clean_text(text):
    # 1. Use ftfy to fix text encoding issues
    text = fix_text(text)

    # 2. Remove the pipe character
    text = text.replace('|', '.')

    # 3. Remove any unusual characters (keeping apostrophes intact)
    text = re.sub(r'[^a-zA-Z0-9\s\'.,!?-]', ' ', text)  # Keeps letters, numbers, common punctuation, and apostrophes

    # 4. Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # 5. Trim leading and trailing spaces
    text = text.strip()

    # 6. Add full stops where missing (simple heuristic)
    if text and text[-1] not in '.!?':
        text += '.'

    # 7. Remove the word "null" from the text
    text = text.replace('null', '')

    return text

# Main function to read the Excel file, clean it, and save as CSV
def main(input_file, output_file):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(input_file)

    # Apply the clean_text function to the 'Concat' column
    if 'Concat' in df.columns:
        df['Concat'] = df['Concat'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    else:
        print("Error: The input Excel file does not have a 'Concat' column.")
        return

    # Save the cleaned DataFrame to a CSV file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_cleanup_concat.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)

