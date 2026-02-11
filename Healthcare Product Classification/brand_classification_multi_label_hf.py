import pandas as pd
import numpy as np
import torch
import sys
from transformers import ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
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

    # 7. Remove the word null from the text
    text = text.replace('null', '') 

    return text

def compute_metrics(pred):
    labels = pred.label_ids
    preds = (pred.predictions >= 0.5).astype(int)  # Binarize the predictions at a threshold of 0.5

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='samples')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main(input_file):
    print("=== Loading Dataset ===")
    df = pd.read_csv(input_file, low_memory=False)
    print("Dataset loaded successfully.")

    # Clean the 'Concat' column
    if 'Concat' in df.columns:
        print("=== Cleaning Dataset ===")
        df['Concat'] = df['Concat'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
        print("Dataset cleaned successfully.")
    else:
        print("Error: The input file does not have a 'Concat' column.")
        return

    # Split the dataset: 65% training, 25% validation, 10% testing
    print("=== Splitting Dataset ===")
    train_set, temp_set = train_test_split(df, test_size=0.35, random_state=42)
    validation_set, test_set = train_test_split(temp_set, test_size=(10/35), random_state=42)
    print(f"Train set: {len(train_set)} samples\nValidation set: {len(validation_set)} samples\nTest set: {len(test_set)} samples")

    # Save each split to CSV files
    train_set.to_csv("training_set.csv", index=False)
    validation_set.to_csv("validation_set_with_labels.csv", index=False)
    test_set.to_csv("test_set_with_labels.csv", index=False)

    # Create unlabeled versions of validation and test sets
    validation_set_unlabeled = validation_set.drop(columns=["Segment", "Sub-Segment", "Target Consumers"])
    test_set_unlabeled = test_set.drop(columns=["Segment", "Sub-Segment", "Target Consumers"])

    # Save the unlabeled versions
    validation_set_unlabeled.to_csv("validation_set_without_labels.csv", index=False)
    test_set_unlabeled.to_csv("test_set_without_labels.csv", index=False)
    print("Datasets saved successfully.")

    # Load tokenizer and model
    print("=== Loading Tokenizer and Model ===")
    model_name = "google/electra-small-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=59, problem_type='multi_label_classification')
    print("Tokenizer and model loaded successfully.")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to {device}")

    # Tokenize the training and validation datasets
    print("=== Tokenizing Dataset ===")
    train_texts = [str(text) for text in train_set['Concat']]  # Ensure all texts are strings
    validation_texts = [str(text) for text in validation_set['Concat']]  # Ensure all texts are strings
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    validation_encodings = tokenizer(validation_texts, truncation=True, padding=True)
    print("Tokenization completed.")

    # Prepare labels for multi-label classification using MultiLabelBinarizer
    print("=== Preparing Labels ===")
    train_labels = train_set[['Segment', 'Sub-Segment', 'Target Consumers']].astype(str).values  # Ensure all labels are strings
    validation_labels = validation_set[['Segment', 'Sub-Segment', 'Target Consumers']].astype(str).values
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_labels).astype(np.float32)
    validation_labels = mlb.transform(validation_labels).astype(np.float32)
    print("Labels prepared successfully.")

    # Convert the datasets to Hugging Face Dataset format
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels.tolist()
    })
    validation_dataset = Dataset.from_dict({
        'input_ids': validation_encodings['input_ids'],
        'attention_mask': validation_encodings['attention_mask'],
        'labels': validation_labels.tolist()
    })
    print("Datasets converted to Hugging Face Dataset format.")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=2,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',     # evaluate at the end of each epoch
        save_strategy='epoch',           # save checkpoint at the end of each epoch
        load_best_model_at_end=True      # load the best model when finished training
    )
    print("Training arguments defined.")

    # Initialize Trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=validation_dataset,     # evaluation dataset
        tokenizer=tokenizer,                 # tokenizer used during training
        compute_metrics=compute_metrics      # function to compute evaluation metrics
    )
    print("Trainer initialized.")

    # Train the model
    print("=== Starting Training ===")
    trainer.train()
    print("Training completed.")

    # Save the model
    print("=== Saving Model ===")
    trainer.save_model("./best_model")
    print("Model saved successfully.")

    # Visualization
    print("=== Visualizing Training Metrics ===")
    training_metrics = trainer.state.log_history

    # Extract metrics from training logs
    training_loss = [log['loss'] for log in training_metrics if 'loss' in log and 'epoch' in log]
    eval_loss = [log['eval_loss'] for log in training_metrics if 'eval_loss' in log and 'epoch' in log]
    eval_accuracy = [log['eval_accuracy'] for log in training_metrics if 'eval_accuracy' in log and 'epoch' in log]
    eval_precision = [log['eval_precision'] for log in training_metrics if 'eval_precision' in log and 'epoch' in log]
    eval_recall = [log['eval_recall'] for log in training_metrics if 'eval_recall' in log and 'epoch' in log]
    eval_f1 = [log['eval_f1'] for log in training_metrics if 'eval_f1' in log and 'epoch' in log]
    epochs = range(1, len(training_loss) + 1)

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(training_loss) + 1), training_loss, 'b-', label='Training Loss')
    plt.plot(range(1, len(eval_loss) + 1), eval_loss, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(eval_accuracy) + 1), eval_accuracy, 'g-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    # Plot Precision and Recall
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(eval_precision) + 1), eval_precision, 'm-', label='Validation Precision')
    plt.plot(range(1, len(eval_recall) + 1), eval_recall, 'c-', label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validation Precision and Recall')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(eval_f1) + 1), eval_f1, 'k-', label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validation F1 Score')
    plt.legend()    

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python brand_classification_multi_label_hf.py <input_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    main(input_file)



