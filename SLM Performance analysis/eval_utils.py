# eval_utils.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
from collections import defaultdict
import json
from pathlib import Path

@dataclass
class DetailedEvalMetrics:
    """Class to store detailed evaluation metrics."""
    accuracy: float
    precision: Dict[str, Dict[str, float]]
    recall: Dict[str, float]
    f1: Dict[str, float]
    confusion_matrix: List[List[float]]
    confidence_stats: Dict[str, float]
    error_analysis: Dict[str, Any]
    length_analysis: Dict[str, float]

class DetailedEvaluator:
    """Handles detailed evaluation of NLI models."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = ['entailment', 'neutral', 'contradiction']
        
    def _compute_confidence_stats(
        self, 
        logits: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistics about model confidence."""
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        
        correct_mask = predictions == labels
        incorrect_mask = ~correct_mask
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'mean_confidence_correct': float(np.mean(confidences[correct_mask])),
            'mean_confidence_incorrect': float(np.mean(confidences[incorrect_mask])),
            'high_confidence_error_rate': float(
                np.mean(incorrect_mask[confidences > 0.9])
            ) if any(confidences > 0.9) else 0.0
        }

    def _analyze_length_impact(
        self, 
        examples: List[Dict], 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Analyze impact of premise/hypothesis length on model performance."""
        premise_lengths = [len(ex['premise'].split()) for ex in examples]
        hypothesis_lengths = [len(ex['hypothesis'].split()) for ex in examples]
        length_diffs = np.array(premise_lengths) - np.array(hypothesis_lengths)
        
        correct_mask = predictions == labels
        
        # Compute correlations
        length_correlations = {
            'premise_length_vs_accuracy': float(np.corrcoef(
                premise_lengths, 
                correct_mask
            )[0, 1]),
            'hypothesis_length_vs_accuracy': float(np.corrcoef(
                hypothesis_lengths, 
                correct_mask
            )[0, 1]),
            'length_difference_vs_accuracy': float(np.corrcoef(
                length_diffs, 
                correct_mask
            )[0, 1])
        }
        
        # Analyze performance by length quartiles
        for lengths, name in [(premise_lengths, 'premise'), 
                            (hypothesis_lengths, 'hypothesis')]:
            quartiles = np.percentile(lengths, [25, 50, 75])
            for i, (lower, upper) in enumerate(zip([0] + list(quartiles), 
                                                 list(quartiles) + [max(lengths)])):
                mask = (np.array(lengths) > lower) & (np.array(lengths) <= upper)
                length_correlations[f'{name}_quartile_{i+1}_accuracy'] = float(
                    np.mean(correct_mask[mask])
                )
        
        return length_correlations

    def _analyze_errors(self, examples, logits, labels, k: int = 10) -> Dict[str, Any]:
        """Detailed error analysis including most confident mistakes."""
        predictions = np.argmax(logits, axis=1)
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        confidences = np.max(probabilities, axis=1)
        
        # Initialize the errors dictionary with proper structure
        errors = {
            'high_confidence_errors': [],
            'error_patterns': {}  # Changed from defaultdict(list) to dict
        }
        
        # Find most confident errors
        error_indices = np.where(predictions != labels)[0]
        error_confidences = confidences[error_indices]
        
        # Convert indices to int before using them
        top_k_error_idx = error_indices[np.argsort(error_confidences)[-k:]].tolist()
        
        # Convert examples to list if it's a Dataset
        examples_list = examples.to_pandas().to_dict('records') if hasattr(examples, 'to_pandas') else examples
        
        for idx in top_k_error_idx:
            errors['high_confidence_errors'].append({
                'premise': examples_list[idx]['premise'],
                'hypothesis': examples_list[idx]['hypothesis'],
                'predicted': self.label_names[predictions[idx]],
                'true': self.label_names[labels[idx]],
                'confidence': float(confidences[idx])
            })
        
        # Analyze error patterns
        for true_label in range(3):
            for pred_label in range(3):
                if true_label != pred_label:
                    mask = (labels == true_label) & (predictions == pred_label)
                    pattern_name = f'{self.label_names[true_label]}_as_{self.label_names[pred_label]}'
                    
                    errors['error_patterns'][pattern_name] = {
                        'count': int(np.sum(mask)),
                        'avg_confidence': float(np.mean(confidences[mask])) if any(mask) else 0.0
                    }
        
        return errors

    def compute_metrics(self, examples, logits, labels):
        """Compute comprehensive evaluation metrics."""
        # Get predicted labels first
        predictions = np.argmax(logits, axis=1)
        
        # Basic metrics
        accuracy = float(np.mean(predictions == labels))
        
        # Per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        per_class_metrics = {}
        for i, label_name in enumerate(self.label_names):
            per_class_metrics[label_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i])
            }
        
        # Confusion matrix
        conf_matrix = np.zeros((3, 3))
        for i in range(len(labels)):
            conf_matrix[labels[i]][predictions[i]] += 1  # Changed from predicted_labels to predictions

        # Convert to list here
        confusion_matrix_list = conf_matrix.tolist()
        
        # Confidence analysis
        confidences = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        confidence_values = np.max(confidences, axis=1)
        
        confidence_stats = {
            'mean_confidence': float(np.mean(confidence_values)),
            'mean_confidence_correct': float(np.mean(confidence_values[predictions == labels])),
            'mean_confidence_incorrect': float(np.mean(confidence_values[predictions != labels])),
            'high_confidence_error_rate': float(
                np.mean((predictions != labels)[confidence_values > 0.9])
            ) if any(confidence_values > 0.9) else 0.0
        }
        
        # Error analysis
        error_analysis = self._analyze_errors(examples, logits, labels)
        
        # Length analysis
        length_analysis = self._analyze_length_impact(examples, predictions, labels)
        
        metrics = DetailedEvalMetrics(
            accuracy=accuracy,
            precision=per_class_metrics,
            recall={label: metrics['recall'] for label, metrics in per_class_metrics.items()},
            f1={label: metrics['f1'] for label, metrics in per_class_metrics.items()},
            confusion_matrix=confusion_matrix_list, 
            confidence_stats=confidence_stats,
            error_analysis=error_analysis,
            length_analysis=length_analysis
        )
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _save_metrics(self, metrics: DetailedEvalMetrics):
        """Save metrics to JSON file."""
        metrics_dict = {
            'accuracy': metrics.accuracy,
            'per_class_metrics': {
                label: {
                    'precision': metrics.precision[label],
                    'recall': metrics.recall[label],
                    'f1': metrics.f1[label]
                }
                for label in self.label_names
            },
            'confusion_matrix': metrics.confusion_matrix,
            'confidence_stats': metrics.confidence_stats,
            'error_analysis': metrics.error_analysis,
            'length_analysis': metrics.length_analysis
        }
        
        with open(self.output_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def analyze_prediction_patterns(
        self, 
        examples: List[Dict], 
        logits: np.ndarray, 
        labels: np.ndarray
    ) -> Dict:
        """Analyze patterns in model predictions."""
        predictions = np.argmax(logits, axis=1)
        patterns = defaultdict(list)
        
        # Analyze consecutive predictions
        for i in range(len(predictions) - 1):
            if predictions[i] == predictions[i + 1]:
                patterns['consecutive_same_predictions'].append({
                    'index': i,
                    'prediction': self.label_names[predictions[i]]
                })
        
        # Analyze prediction transitions
        transitions = np.zeros((3, 3))
        for i in range(len(predictions) - 1):
            transitions[predictions[i], predictions[i + 1]] += 1
        
        patterns['prediction_transitions'] = transitions.tolist()
        
        # Save patterns
        with open(self.output_dir / 'prediction_patterns.json', 'w') as f:
            json.dump(patterns, f, indent=2)
        
        return patterns