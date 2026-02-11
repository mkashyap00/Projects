# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import pandas as pd

class ResultsVisualizer:
    """Creates visualizations for model evaluation results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = ['entailment', 'neutral', 'contradiction']
        # Set style for all plots
        sns.set_style("whitegrid")
        
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray,
        title: str = "Confusion Matrix"
    ):
        """Plot confusion matrix with percentages."""
        plt.figure(figsize=(10, 8))
        
        # Convert to percentages
        conf_matrix_percent = (
            confusion_matrix.astype('float') / 
            confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
        )
        
        sns.heatmap(
            conf_matrix_percent,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confidence_distribution(
        self, 
        confidence_stats: Dict[str, float]
    ):
        """Plot confidence distribution for correct vs incorrect predictions."""
        plt.figure(figsize=(10, 6))
        
        data = [
            ('Correct', confidence_stats['mean_confidence_correct']),
            ('Incorrect', confidence_stats['mean_confidence_incorrect'])
        ]
        
        colors = ['#2ecc71', '#e74c3c']
        bars = plt.bar([x[0] for x in data], [x[1] for x in data], color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )
        
        plt.title('Mean Confidence: Correct vs Incorrect Predictions')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_length_impact(
        self, 
        length_analysis: Dict[str, float]
    ):
        """Plot impact of sequence length on model performance."""
        plt.figure(figsize=(12, 6))
        
        # Extract quartile accuracies
        premise_quartiles = [
            length_analysis[f'premise_quartile_{i}_accuracy']
            for i in range(1, 5)
        ]
        hypothesis_quartiles = [
            length_analysis[f'hypothesis_quartile_{i}_accuracy']
            for i in range(1, 5)
        ]
        
        x = np.arange(4)
        width = 0.35
        
        plt.bar(x - width/2, premise_quartiles, width, label='Premise')
        plt.bar(x + width/2, hypothesis_quartiles, width, label='Hypothesis')
        
        plt.xlabel('Length Quartile')
        plt.ylabel('Accuracy')
        plt.title('Model Performance by Sequence Length')
        plt.xticks(x, ['Q1', 'Q2', 'Q3', 'Q4'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'length_impact.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bias_analysis(
        self, 
        bias_results: Dict
    ):
        """Create visualizations for bias analysis results."""
        # Hypothesis-only performance
        plt.figure(figsize=(10, 6))
        per_class_acc = bias_results['hypothesis_only']['per_class_accuracies']
        
        plt.bar(
            per_class_acc.keys(),
            per_class_acc.values(),
            color='skyblue'
        )
        plt.axhline(
            y=0.33,
            color='r',
            linestyle='--',
            label='Random Chance'
        )
        plt.title('Hypothesis-Only Classification Performance')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_only_bias.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Lexical overlap analysis
        if 'lexical_overlap' in bias_results:
            plt.figure(figsize=(10, 6))
            overlap_by_class = bias_results['lexical_overlap']['average_overlap_by_class']
            
            plt.bar(
                overlap_by_class.keys(),
                overlap_by_class.values(),
                color='lightgreen'
            )
            plt.title('Average Lexical Overlap by Class')
            plt.ylabel('Overlap Ratio')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'lexical_overlap_bias.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_error_patterns(
        self, 
        error_analysis: Dict[str, List]
    ):
        """Visualize common error patterns."""
        patterns = pd.DataFrame(error_analysis['error_patterns']).T
        
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        patterns['count'].plot(kind='bar')
        plt.title('Error Counts by Pattern')
        plt.xticks(rotation=45, ha='right')
        
        ax2 = plt.subplot(1, 2, 2)
        patterns['avg_confidence'].plot(kind='bar')
        plt.title('Average Confidence in Errors')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_all_visualizations(
        self, 
        eval_metrics_path: str,
        bias_results_path: Optional[str] = None
    ):
        """Create all visualizations from saved metrics."""
        # Load evaluation metrics
        with open(eval_metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Create basic evaluation visualizations
        self.plot_confusion_matrix(
            np.array(metrics['confusion_matrix'])
        )
        self.plot_confidence_distribution(
            metrics['confidence_stats']
        )
        self.plot_length_impact(
            metrics['length_analysis']
        )
        self.plot_error_patterns(
            metrics['error_analysis']
        )
        
        # Create bias visualizations if available
        if bias_results_path:
            with open(bias_results_path, 'r') as f:
                bias_results = json.load(f)
            self.plot_bias_analysis(bias_results)
        
        # Create summary figure
        self.create_summary_figure(metrics)

    def create_summary_figure(
        self, 
        metrics: Dict
    ):
        """Create a summary figure with key metrics."""
        plt.figure(figsize=(15, 10))
        
        # Overall accuracy
        plt.subplot(2, 2, 1)
        plt.bar(['Overall Accuracy'], [metrics['accuracy']], color='blue')
        plt.title('Overall Accuracy')
        plt.ylim(0, 1)
        
        # Per-class metrics
        plt.subplot(2, 2, 2)
        class_metrics = pd.DataFrame(metrics['per_class_metrics']).T
        class_metrics[['precision', 'recall', 'f1']].plot(kind='bar')
        plt.title('Per-Class Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary.png', dpi=300, bbox_inches='tight')
        plt.close()