import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import ElectraConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class ProductDataset(Dataset):
    """
    Dataset class for product data
    """
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
    
def get_label_or_unknown(label):
    """Convert label to string, handling NaN values"""
    if pd.isna(label):
        return 'Unknown'
    return str(label)

def calculate_metrics_per_category(confidence_scores, label_info):
    """Calculate confidence metrics for each label category"""
    metrics = {}
    start_idx = 0
    
    for category, labels in {
        'segments': label_info['segment_classes'],
        'subsegments': label_info['subsegment_classes'],
        'consumers': label_info['target_consumer_classes']
    }.items():
        scores = confidence_scores[:, start_idx:start_idx + len(labels)]
        metrics[category] = {
            'mean_confidence': float(np.mean(scores)),
            'std_confidence': float(np.std(scores)),
            'median_confidence': float(np.median(scores))
        }
        start_idx += len(labels)
    
    return metrics

def plot_confidence_distributions(confidence_scores, label_info):
    """Plot confidence score distributions for each label category"""
    os.makedirs('plots', exist_ok=True)
    
    categories = {
        'segments': label_info['segment_classes'],
        'subsegments': label_info['subsegment_classes'],
        'consumers': label_info['target_consumer_classes']
    }
    
    start_idx = 0
    for category, labels in categories.items():
        plt.figure(figsize=(10, 6))
        scores = confidence_scores[:, start_idx:start_idx + len(labels)]
        sns.boxplot(data=pd.DataFrame(scores, columns=labels))
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Confidence Distribution - {category.capitalize()}')
        plt.tight_layout()
        plt.savefig(f'plots/{category}_confidence_dist.png')
        plt.close()
        start_idx += len(labels)

def predict_test_data(unlabeled_test_path, labeled_test_path, model_path, device):
    """
    Make predictions on unseen test data and evaluate against true labels
    Args:
        unlabeled_test_path: Path to test CSV file without labels
        labeled_test_path: Path to test CSV file with labels for evaluation
        model_path: Path to saved model checkpoint
        device: torch device
    """
    # Load data files
    logger.info(f"Loading unlabeled test data from {unlabeled_test_path}")
    df_test_unlabeled = pd.read_csv(unlabeled_test_path)
    logger.info(f"Unlabeled data shape: {df_test_unlabeled.shape}")
    
    logger.info(f"Loading labeled test data from {labeled_test_path}")
    df_test_labeled = pd.read_csv(labeled_test_path)
    logger.info(f"Labeled data shape: {df_test_labeled.shape}")
    
    # Check number of rows
    if len(df_test_unlabeled) != len(df_test_labeled):
        raise ValueError(f"Files have different number of rows! Unlabeled: {len(df_test_unlabeled)}, Labeled: {len(df_test_labeled)}")
    
    # Check for NaN values
    unlabeled_nan_count = df_test_unlabeled['Concat'].isna().sum()
    labeled_nan_count = df_test_labeled['Concat'].isna().sum()
    
    if unlabeled_nan_count > 0 or labeled_nan_count > 0:
        logger.warning(f"Found NaN values - Unlabeled: {unlabeled_nan_count}, Labeled: {labeled_nan_count}")
        
    # Convert to string and fill NaN
    df_test_unlabeled['Concat'] = df_test_unlabeled['Concat'].fillna('').astype(str)
    df_test_labeled['Concat'] = df_test_labeled['Concat'].fillna('').astype(str)
    
    # Check content matching
    logger.info("Checking content matching between files...")
    content_match = all(df_test_unlabeled['Concat'] == df_test_labeled['Concat'])
    if not content_match:
        # Try to identify differences
        for idx, (unlabeled_text, labeled_text) in enumerate(zip(df_test_unlabeled['Concat'], df_test_labeled['Concat'])):
            if unlabeled_text != labeled_text:
                logger.error(f"\nMismatch at row {idx}:")
                logger.error(f"Unlabeled text: {unlabeled_text[:min(100, len(unlabeled_text))]}")
                logger.error(f"Labeled text: {labeled_text[:min(100, len(labeled_text))]}")
                break
        
        logger.info("\nAttempting to sort texts for comparison...")
        # Try sorting both dataframes by text
        sorted_unlabeled = df_test_unlabeled.sort_values('Concat').reset_index(drop=True)
        sorted_labeled = df_test_labeled.sort_values('Concat').reset_index(drop=True)
        
        if all(sorted_unlabeled['Concat'] == sorted_labeled['Concat']):
            logger.info("Files contain the same content but in different order. Proceeding with prediction...")
            # Use the order from unlabeled file but keep track of original indices
            df_test_labeled = df_test_labeled.set_index('Concat').reindex(index=df_test_unlabeled['Concat']).reset_index()
        else:
            raise ValueError("Text content doesn't match between files even after sorting!")
    
    # Verify that the text content matches between files
    if not all(df_test_unlabeled['Concat'] == df_test_labeled['Concat']):
        raise ValueError("Text content doesn't match between labeled and unlabeled files!")
    
    # Load the saved model and label info
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    label_info = checkpoint['label_info']
    
    # Initialize model with same configuration from saved checkpoint
    config = ElectraConfig.from_pretrained(
        'google/electra-base-discriminator',
        num_labels=63,  # This matches our trained model
        problem_type="multi_label_classification"
    )
    
    model = ElectraForSequenceClassification.from_pretrained(
        'google/electra-base-discriminator',
        config=config
    )
    
    # Load saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    
    # Create test dataset and dataloader
    test_texts = df_test_unlabeled['Concat'].values
    test_dataset = ProductDataset(test_texts, np.zeros((len(test_texts), model.num_labels)), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = []
    confidence_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_probs = torch.sigmoid(outputs.logits)
            pred_labels = (pred_probs > 0.5).cpu().numpy()
            predictions.extend(pred_labels)
            confidence_scores.extend(pred_probs.cpu().numpy())
    
    predictions = np.array(predictions)
    confidence_scores = np.array(confidence_scores)
    
    # Calculate detailed metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics_per_category(confidence_scores, label_info)
    
    # Create visualizations
    logger.info("\nGenerating confidence distribution plots...")
    plot_confidence_distributions(confidence_scores, label_info)
    logger.info("Plots saved in 'plots' directory")
    
    # Convert predictions to human-readable labels with confidence scores
    n_segments = len(label_info['segment_classes'])
    n_subsegments = len(label_info['subsegment_classes'])
    
    predicted_labels = []
    for pred, conf in zip(predictions, confidence_scores):
        segment_idx = np.where(pred[:n_segments] == 1)[0]
        subsegment_idx = np.where(pred[n_segments:n_segments+n_subsegments] == 1)[0]
        consumer_idx = np.where(pred[n_segments+n_subsegments:] == 1)[0]
        
        pred_dict = {
            'segments': [label_info['segment_classes'][i] for i in segment_idx],
            'segment_confidences': [conf[i] for i in segment_idx],
            'subsegments': [label_info['subsegment_classes'][i] for i in subsegment_idx],
            'subsegment_confidences': [conf[n_segments + i] for i in subsegment_idx],
            'consumers': [label_info['target_consumer_classes'][i] for i in consumer_idx],
            'consumer_confidences': [conf[n_segments + n_subsegments + i] for i in consumer_idx]
        }
        predicted_labels.append(pred_dict)
    
    # Create output dataframe
    df_output = pd.DataFrame({
        'Text': test_texts,
        'Predicted_Segments': [pred['segments'] for pred in predicted_labels],
        'Segment_Confidences': [pred['segment_confidences'] for pred in predicted_labels],
        'Predicted_Subsegments': [pred['subsegments'] for pred in predicted_labels],
        'Subsegment_Confidences': [pred['subsegment_confidences'] for pred in predicted_labels],
        'Predicted_Consumers': [pred['consumers'] for pred in predicted_labels],
        'Consumer_Confidences': [pred['consumer_confidences'] for pred in predicted_labels],
        'True_Segments': df_test_labeled['Segment'].values,
        'True_Subsegments': df_test_labeled['Sub-Segment'].values,
        'True_Consumers': df_test_labeled['Target Consumers'].values
    })
    
    # Calculate comparison metrics
    logger.info("\nCalculating comparison metrics...")
    comparison_metrics = {
        'segments': {
            'accuracy': accuracy_score(
                [get_label_or_unknown(x) for x in df_test_labeled['Segment'].values],
                [pred['segments'][0] if pred['segments'] else 'Unknown' for pred in predicted_labels]
            ),
            'precision_recall_f1': precision_recall_fscore_support(
                [get_label_or_unknown(x) for x in df_test_labeled['Segment'].values],
                [pred['segments'][0] if pred['segments'] else 'Unknown' for pred in predicted_labels],
                average='weighted'
            )
        },
        'subsegments': {
            'accuracy': accuracy_score(
                [get_label_or_unknown(x) for x in df_test_labeled['Sub-Segment'].values],
                [pred['subsegments'][0] if pred['subsegments'] else 'Unknown' for pred in predicted_labels]
            ),
            'precision_recall_f1': precision_recall_fscore_support(
                [get_label_or_unknown(x) for x in df_test_labeled['Sub-Segment'].values],
                [pred['subsegments'][0] if pred['subsegments'] else 'Unknown' for pred in predicted_labels],
                average='weighted'
            )
        },
        'consumers': {
            'accuracy': accuracy_score(
                [get_label_or_unknown(x) for x in df_test_labeled['Target Consumers'].values],
                [pred['consumers'][0] if pred['consumers'] else 'Unknown' for pred in predicted_labels]
            ),
            'precision_recall_f1': precision_recall_fscore_support(
                [get_label_or_unknown(x) for x in df_test_labeled['Target Consumers'].values],
                [pred['consumers'][0] if pred['consumers'] else 'Unknown' for pred in predicted_labels],
                average='weighted'
            )
        }
    }
    
    # Log comparison metrics
    logger.info("\nComparison with True Labels:")
    for category, metrics_dict in comparison_metrics.items():
        logger.info(f"\n{category.capitalize()}:")
        logger.info(f"Accuracy: {metrics_dict['accuracy']:.4f}")
        precision, recall, f1, _ = metrics_dict['precision_recall_f1']
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
    
    # Save predictions and metrics
    output_path = 'test_predictions_with_true_labels.csv'
    metrics_path = 'test_metrics_with_comparison.json'
    
    df_output.to_csv(output_path, index=False)
    with open(metrics_path, 'w') as f:
        json.dump({
            'confidence_metrics': metrics,
            'comparison_metrics': comparison_metrics
        }, f, indent=2)
    
    logger.info(f"\nPredictions saved to {output_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return df_output, metrics, comparison_metrics

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Make predictions on test data')
    parser.add_argument('unlabeled_test_path', type=str, help='Path to test CSV file without labels')
    parser.add_argument('labeled_test_path', type=str, help='Path to test CSV file with labels')
    parser.add_argument('model_path', type=str, help='Path to saved model checkpoint')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictions, metrics, comparison_metrics = predict_test_data(
        args.unlabeled_test_path, 
        args.labeled_test_path,
        args.model_path, 
        device
    )