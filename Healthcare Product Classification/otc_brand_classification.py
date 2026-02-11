import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class ProductDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

def parse_department_hierarchy(department_str):
    """Parse hierarchical department string into levels"""
    if pd.isna(department_str):
        return []
    return [level.strip() for level in str(department_str).split('>')]

def prepare_data(file_path):
    """Load and prepare the dataset with hierarchical label structure"""
    logger.info(f"Loading data from {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Parse department hierarchy
    logger.info("Processing Department hierarchy")
    df['Department_Levels'] = df['Department'].apply(parse_department_hierarchy)
    
    # Analyze department hierarchy
    max_depth = df['Department_Levels'].apply(len).max()
    logger.info(f"Maximum department hierarchy depth: {max_depth}")
    
    # Get unique values at each level
    dept_levels = {}
    for level in range(max_depth):
        unique_values = set()
        for dept_list in df['Department_Levels']:
            if len(dept_list) > level:
                unique_values.add(dept_list[level])
        dept_levels[f"Level_{level}"] = unique_values
        logger.info(f"Department Level {level} unique values: {len(unique_values)}")
    
    # Extract target variables
    target_columns = ['Segment', 'Sub-Segment', 'Target Consumers']
    
    # Create MultiLabelBinarizer for each target
    mlbs = {}
    label_encodings = {}
    
    # Process Segment first
    logger.info("\nProcessing Segment labels")
    segment_mlb = MultiLabelBinarizer()
    segment_labels = df['Segment'].fillna('Unknown').apply(lambda x: [x])
    segment_encoded = segment_mlb.fit_transform(segment_labels)
    mlbs['Segment'] = segment_mlb
    label_encodings['Segment'] = segment_encoded
    logger.info(f"Unique Segments: {len(segment_mlb.classes_)}")
    
    # Process Sub-Segment with relationship to Segment
    logger.info("Processing Sub-Segment labels")
    subsegment_mlb = MultiLabelBinarizer()
    subsegment_labels = df['Sub-Segment'].fillna('Unknown').apply(lambda x: [x])
    subsegment_encoded = subsegment_mlb.fit_transform(subsegment_labels)
    mlbs['Sub-Segment'] = subsegment_mlb
    label_encodings['Sub-Segment'] = subsegment_encoded
    logger.info(f"Unique Sub-Segments: {len(subsegment_mlb.classes_)}")
    
    # Process Target Consumers
    logger.info("\nProcessing Target Consumers labels")
    target_mlb = MultiLabelBinarizer()
    target_labels = df['Target Consumers'].fillna('Unknown').apply(lambda x: [x])
    target_encoded = target_mlb.fit_transform(target_labels)
    mlbs['Target Consumers'] = target_mlb
    label_encodings['Target Consumers'] = target_encoded
    logger.info(f"Unique Target Consumers: {len(target_mlb.classes_)}")
    
    # Combine all encoded labels
    all_labels = np.hstack([label_encodings[col] for col in target_columns])
    logger.info(f"\nTotal number of encoded labels: {all_labels.shape[1]}")
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['Concat'].values, all_labels,
        test_size=0.35,  # 65% for training
        random_state=RANDOM_SEED
    )
    
    # Split temp into validation and test
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.286,  # 10% of total for test, 25% for validation
        random_state=RANDOM_SEED
    )
    
    logger.info(f"\nTrain size: {len(train_texts)}, Validation size: {len(val_texts)}, Test size: {len(test_texts)}")
    
    # Save the label relationships and department information
    label_info = {
        'segment_classes': segment_mlb.classes_,
        'subsegment_classes': subsegment_mlb.classes_,
        'target_consumer_classes': target_mlb.classes_,
        'department_levels': dept_levels,
        'max_department_depth': max_depth
    }
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), mlbs, label_info

def create_data_loaders(train_data, val_data, test_data, tokenizer, batch_size=16):
    """Create DataLoaders for training, validation and testing"""
    train_dataset = ProductDataset(train_data[0], train_data[1], tokenizer)
    val_dataset = ProductDataset(val_data[0], val_data[1], tokenizer)
    test_dataset = ProductDataset(test_data[0], test_data[1], tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    predictions = []
    actual_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            predictions.extend(torch.sigmoid(outputs.logits).cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    # Calculate metrics
    predictions_binary = (predictions > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels, predictions_binary,
        average='weighted',
        zero_division=0
    )
    accuracy = accuracy_score(actual_labels, predictions_binary)
    
    # Calculate per-label metrics
    per_label_precision, per_label_recall, per_label_f1, _ = precision_recall_fscore_support(
        actual_labels, predictions_binary,
        average=None,
        zero_division=0
    )
    
    # Calculate confusion matrix per label
    n_labels = actual_labels.shape[1]
    confusion_matrices = []
    for i in range(n_labels):
        cm = confusion_matrix(actual_labels[:, i], predictions_binary[:, i])
        confusion_matrices.append(cm)
    
    return {
        'loss': total_loss / len(data_loader),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'per_label_precision': per_label_precision,
        'per_label_recall': per_label_recall,
        'per_label_f1': per_label_f1,
        'confusion_matrices': confusion_matrices
    }

def get_prediction_examples(model, data_loader, texts, mlbs, label_info, device, n_examples=10):
    """Get examples of correct and incorrect predictions"""
    model.eval()
    correct_examples = []
    incorrect_examples = []
    
    def get_label_names(binary_array, start_idx, label_classes):
        """Convert binary array to human-readable label names"""
        return [label_classes[i] for i, val in enumerate(binary_array[start_idx:start_idx + len(label_classes)]) if val == 1]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits)
            predictions = (predictions > 0.5).cpu().numpy()
            actual = labels.cpu().numpy()
            
            batch_start_idx = batch_idx * data_loader.batch_size
            batch_texts = texts[batch_start_idx:batch_start_idx + len(predictions)]
            
            for idx in range(len(predictions)):
                if len(correct_examples) >= n_examples and len(incorrect_examples) >= n_examples:
                    break
                    
                pred = predictions[idx]
                true = actual[idx]
                text = batch_texts[idx]
                
                n_segments = len(label_info['segment_classes'])
                n_subsegments = len(label_info['subsegment_classes'])
                
                example = {
                    'text': text,
                    'predicted': {
                        'segments': get_label_names(pred, 0, label_info['segment_classes']),
                        'subsegments': get_label_names(pred, n_segments, label_info['subsegment_classes']),
                        'consumers': get_label_names(pred, n_segments + n_subsegments, 
                                                   label_info['target_consumer_classes'])
                    },
                    'actual': {
                        'segments': get_label_names(true, 0, label_info['segment_classes']),
                        'subsegments': get_label_names(true, n_segments, label_info['subsegment_classes']),
                        'consumers': get_label_names(true, n_segments + n_subsegments,
                                                   label_info['target_consumer_classes'])
                    }
                }
                
                if np.array_equal(pred, true) and len(correct_examples) < n_examples:
                    correct_examples.append(example)
                elif not np.array_equal(pred, true) and len(incorrect_examples) < n_examples:
                    incorrect_examples.append(example)
            
            if len(correct_examples) >= n_examples and len(incorrect_examples) >= n_examples:
                break
                
    return correct_examples, incorrect_examples

def log_prediction_examples(correct_examples, incorrect_examples):
    """Log examples of correct and incorrect predictions"""
    logger.info("\n" + "="*20 + " PREDICTION EXAMPLES " + "="*20)
    
    logger.info("\nCORRECT PREDICTIONS:")
    for i, example in enumerate(correct_examples, 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"Text: {example['text'][:200]}...")
        logger.info("Predicted (Correct):")
        logger.info(f"  Segments: {example['predicted']['segments']}")
        logger.info(f"  Sub-segments: {example['predicted']['subsegments']}")
        logger.info(f"  Target Consumers: {example['predicted']['consumers']}")
    
    logger.info("\nINCORRECT PREDICTIONS:")
    for i, example in enumerate(incorrect_examples, 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"Text: {example['text'][:200]}...")
        logger.info("Predicted (Incorrect):")
        logger.info(f"  Segments: {example['predicted']['segments']}")
        logger.info(f"  Sub-segments: {example['predicted']['subsegments']}")
        logger.info(f"  Target Consumers: {example['predicted']['consumers']}")
        logger.info("Actual:")
        logger.info(f"  Segments: {example['actual']['segments']}")
        logger.info(f"  Sub-segments: {example['actual']['subsegments']}")
        logger.info(f"  Target Consumers: {example['actual']['consumers']}")
    
    logger.info("\n" + "="*50)

def plot_metrics(training_stats):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot([x['epoch'] for x in training_stats], 
             [x['train_loss'] for x in training_stats], 
             label='Training Loss')
    ax1.plot([x['epoch'] for x in training_stats], 
             [x['val_loss'] for x in training_stats], 
             label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot metrics
    ax2.plot([x['epoch'] for x in training_stats], 
             [x['val_accuracy'] for x in training_stats], 
             label='Accuracy')
    ax2.plot([x['epoch'] for x in training_stats], 
             [x['val_f1'] for x in training_stats], 
             label='F1 Score')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main(dataset_path):
    """
    Main execution function
    Args:
        dataset_path (str): Path to the CSV dataset file
    """
    # Prepare data
    logger.info("Preparing data...")
    train_data, val_data, test_data, mlbs, label_info = prepare_data(dataset_path)
    
    # Calculate total number of labels
    total_labels = train_data[1].shape[1]
    logger.info(f"Total number of labels: {total_labels}")
    
    # Initialize tokenizer and model
    logger.info("Initializing model...")
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    model = ElectraForSequenceClassification.from_pretrained(
        'google/electra-base-discriminator',
        num_labels=total_labels,
        problem_type="multi_label_classification",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2
    )
    model.to(device)
    
    # Create data loaders with increased batch size
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, tokenizer, batch_size=16
    )
    
    # Training parameters
    num_epochs = 10  # increased from 5
    
    # Initialize optimizer with lower learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Create scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    training_stats = []
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")
        
        # Evaluate
        logger.info("Running validation...")
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Validation Metrics: Accuracy={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")
        
        # Save stats
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1']
        })
        
        # Save checkpoint if best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            logger.info(f"New best F1 score: {best_val_f1:.4f} - Saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'label_info': label_info,
                'mlbs': mlbs
            }, f'{checkpoint_dir}/best_model.pt')
        
        # Plot metrics
        plot_metrics(training_stats)
    
    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_metrics = evaluate(model, test_loader, device)
    logger.info("Final Test Metrics:")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
    
    # Get prediction examples
    logger.info("Gathering prediction examples...")
    correct_examples, incorrect_examples = get_prediction_examples(
        model, test_loader, test_data[0], mlbs, label_info, device
    )
    log_prediction_examples(correct_examples, incorrect_examples)

if __name__ == "__main__":
    import argparse
    from transformers import get_linear_schedule_with_warmup
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Multi-label classification for product data')
    parser.add_argument('dataset_path', type=str, help='Path to the CSV dataset file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with dataset path
    main(args.dataset_path)