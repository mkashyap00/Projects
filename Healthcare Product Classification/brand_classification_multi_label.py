import pandas as pd
import re
import sys
from ftfy import fix_text
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os

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

    # 7. Remove the word "null" from the text
    text = text.replace('null', '')

    return text

# Move CustomDataset class to top level to avoid pickling issues with multiprocessing
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)

# Main function to split the dataset and train the model
def main(input_file):
    # Data Loading
    print("\n=== Loading Dataset ===")
    print(f"Reading from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Dataset loaded with {len(df)} rows")

    # # Clean the 'Concat' column
    # if 'Concat' in df.columns:
    #     df['Concat'] = df['Concat'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    # else:
    #     print("Error: The input file does not have a 'Concat' column.")
    #     return

    # Data Splitting
    print("\n=== Splitting Dataset ===")
    train_set, temp_set = train_test_split(df, test_size=0.35, random_state=42)
    validation_set, test_set = train_test_split(temp_set, test_size=(10/35), random_state=42)
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(validation_set)} samples")
    print(f"Test set: {len(test_set)} samples")

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

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    model_name = "google/electra-small-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type='multi_label_classification')

    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    print("Model loaded successfully")

    # Tokenize the training dataset
    def tokenize_function(examples):
        return tokenizer(examples['Concat'], padding='max_length', truncation=True)

    # Tokenize the datasets
    train_dataset = pd.read_csv("training_set.csv")
    train_dataset_tokenized = train_dataset.apply(lambda x: tokenize_function(x), axis=1)

    # Prepare labels for multi-label classification using MultiLabelBinarizer
    train_labels = train_dataset[['Segment', 'Sub-Segment', 'Target Consumers']].values
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_labels).astype(np.float32).tolist()     # Convert labels to binary format for multi-label classification

    # Prepare Dataset for DataLoader
    train_encodings = tokenizer(list(train_dataset['Concat']), truncation=True, padding=True)
    train_dataset_final = CustomDataset(train_encodings, train_labels)

    # DataLoader with num_workers for faster data loading
    train_loader = DataLoader(train_dataset_final, batch_size=16, shuffle=True, num_workers=0)  # Set num_workers to 0 to avoid multiprocessing issues on Windows
    validation_loader = DataLoader(train_dataset_final, batch_size=16, num_workers=0)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Mixed precision training
    scaler = GradScaler('cuda')
    num_epochs = 2  # Reduced epochs for testing purposes

    # Checkpoint directory
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Training Setup
    print("\n=== Training Configuration ===")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Training loop with mixed precision and optimizer
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Mixed precision
            with autocast():
                outputs = model(inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        print(f"Training loss: {total_loss / len(train_loader):.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Early stop condition for testing
        if epoch == 0:  # Stop after the first epoch for testing purposes
            print("Stopping early for testing purposes...")
            break

    # Extract validation metrics from training logs
    training_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    eval_accuracy = [log['eval_accuracy'] for log in trainer.state.log_history if 'eval_accuracy' in log]
    eval_precision = [log['eval_precision'] for log in trainer.state.log_history if 'eval_precision' in log]
    eval_recall = [log['eval_recall'] for log in trainer.state.log_history if 'eval_recall' in log]
    eval_f1 = [log['eval_f1'] for log in trainer.state.log_history if 'eval_f1' in log]
    epochs = range(1, len(training_loss) + 1)

    # Visualization of Training and Validation Metrics
    plt.figure(figsize=(12, 8))

    # Plot Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_loss, 'b-', label='Training Loss')
    plt.plot(epochs, eval_loss, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, eval_accuracy, 'g-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    # Plot Precision and Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, eval_precision, 'm-', label='Validation Precision')
    plt.plot(epochs, eval_recall, 'c-', label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validation Precision and Recall')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, eval_f1, 'k-', label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_metrics.png')
    plt.show()

    # Confusion Matrix Visualization
    val_labels = validation_set[['Segment', 'Sub-Segment', 'Target Consumers']].values
    val_labels = mlb.transform(val_labels).astype(np.float32)  # Convert validation labels to binary format
    val_encodings = tokenizer(list(validation_set['Concat']), truncation=True, padding=True, return_tensors='pt').to(device)
    val_outputs = model(**val_encodings)
    val_preds = (torch.sigmoid(val_outputs.logits) > 0.5).cpu().numpy()

    cm = confusion_matrix(val_labels.argmax(axis=1), val_preds.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sample_cleanup_concat.py <input_file>")
    else:
        input_file = sys.argv[1]
        main(input_file)    
