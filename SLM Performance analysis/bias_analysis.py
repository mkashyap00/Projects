import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict
import spacy
from tqdm import tqdm

class BiasAnalyzer:
    """Analyzes different types of biases in NLI models and datasets."""
    
    def __init__(self, model_path: str, output_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Initialize spaCy for lexical overlap analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Bias metrics storage
        self.bias_metrics = defaultdict(dict)

    def analyze_hypothesis_only(self, dataset) -> Dict:
        """
        Analyze model performance using only hypothesis inputs.
        This helps detect if the model is learning shortcuts based on hypothesis alone.
        """
        print("Analyzing hypothesis-only bias...")
        
        # Create hypothesis-only inputs
        hypothesis_inputs = self.tokenizer(
            [example['hypothesis'] for example in dataset],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Run in small batches to manage memory
        batch_size = 32
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = {k: v[i:i+batch_size].to(self.device) 
                        for k, v in hypothesis_inputs.items()}
                outputs = self.model(**batch)
                predictions = outputs.logits.cpu()
                all_predictions.append(predictions)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        predicted_labels = all_predictions.argmax(dim=-1).numpy()
        
        # Calculate hypothesis-only accuracy
        true_labels = [example['label'] for example in dataset]
        hypothesis_only_acc = (predicted_labels == true_labels).mean()
        
        # Analyze per-class performance
        class_accuracies = {}
        for label in set(true_labels):
            mask = np.array(true_labels) == label
            class_accuracies[f"class_{label}"] = (
                predicted_labels[mask] == np.array(true_labels)[mask]
            ).mean()
        
        results = {
            "hypothesis_only_accuracy": float(hypothesis_only_acc),
            "per_class_accuracies": class_accuracies
        }
        
        self.bias_metrics["hypothesis_only"] = results
        return results

    def analyze_length_bias(self, dataset) -> Dict:
        """
        Analyze correlation between example lengths and predictions.
        """
        print("Analyzing length bias...")
        
        length_metrics = {
            "premise_lengths": [],
            "hypothesis_lengths": [],
            "length_differences": [],
            "predictions": [],
            "labels": []
        }
        
        # Calculate lengths and collect predictions
        for example in dataset:
            premise_tokens = self.nlp(example['premise'])
            hypothesis_tokens = self.nlp(example['hypothesis'])
            
            length_metrics["premise_lengths"].append(len(premise_tokens))
            length_metrics["hypothesis_lengths"].append(len(hypothesis_tokens))
            length_metrics["length_differences"].append(
                len(premise_tokens) - len(hypothesis_tokens)
            )
            length_metrics["labels"].append(example['label'])
        
        # Convert to numpy arrays for correlation analysis
        for key in length_metrics:
            length_metrics[key] = np.array(length_metrics[key])
        
        # Calculate correlations
        correlations = {
            "premise_length_vs_label": float(np.corrcoef(
                length_metrics["premise_lengths"], 
                length_metrics["labels"]
            )[0, 1]),
            "hypothesis_length_vs_label": float(np.corrcoef(
                length_metrics["hypothesis_lengths"], 
                length_metrics["labels"]
            )[0, 1]),
            "length_difference_vs_label": float(np.corrcoef(
                length_metrics["length_differences"], 
                length_metrics["labels"]
            )[0, 1])
        }
        
        self.bias_metrics["length"] = correlations
        return correlations

    def analyze_lexical_overlap(self, dataset) -> Dict:
        """
        Analyze the impact of lexical overlap between premise and hypothesis.
        """
        print("Analyzing lexical overlap bias...")
        
        overlap_metrics = []
        
        for example in tqdm(dataset, desc="Computing lexical overlap"):
            # Get content words from premise and hypothesis
            premise_doc = self.nlp(example['premise'])
            hypothesis_doc = self.nlp(example['hypothesis'])
            
            premise_content = set(
                token.text.lower() for token in premise_doc 
                if not token.is_stop and not token.is_punct
            )
            hypothesis_content = set(
                token.text.lower() for token in hypothesis_doc 
                if not token.is_stop and not token.is_punct
            )
            
            # Calculate overlap ratio
            if len(hypothesis_content) > 0:
                overlap_ratio = len(
                    premise_content.intersection(hypothesis_content)
                ) / len(hypothesis_content)
            else:
                overlap_ratio = 0
                
            overlap_metrics.append({
                "overlap_ratio": overlap_ratio,
                "label": example['label']
            })
        
        # Analyze correlation between overlap and labels
        overlap_ratios = np.array([m["overlap_ratio"] for m in overlap_metrics])
        labels = np.array([m["label"] for m in overlap_metrics])
        
        correlation = float(np.corrcoef(overlap_ratios, labels)[0, 1])
        
        # Calculate average overlap by class
        overlap_by_class = {}
        for label in set(labels):
            mask = labels == label
            overlap_by_class[f"class_{label}"] = float(
                overlap_ratios[mask].mean()
            )
        
        results = {
            "overlap_label_correlation": correlation,
            "average_overlap_by_class": overlap_by_class
        }
        
        self.bias_metrics["lexical_overlap"] = results
        return results
    
    def run_full_analysis(self, dataset) -> Dict:
        """
        Run all bias analyses and save results.
        """
        print("Running full bias analysis...")
        
        self.analyze_hypothesis_only(dataset)
        self.analyze_length_bias(dataset)
        self.analyze_lexical_overlap(dataset)
        
        # Save all results
        output_path = self.output_dir / "bias_analysis_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.bias_metrics, f, indent=2)
        
        return self.bias_metrics

    def get_most_biased_examples(self, dataset, top_k: int = 10) -> Dict:
        """
        Identify examples that exhibit the strongest bias patterns.
        """
        biased_examples = {
            "hypothesis_only": [],
            "length_bias": [],
            "high_overlap": []
        }
        
        # Find examples with high hypothesis-only confidence
        inputs = self.tokenizer(
            [example['hypothesis'] for example in dataset],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))
            confidences = torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0].cpu()
        
        # Sort examples by confidence
        hypothesis_only_examples = list(zip(dataset, confidences))
        hypothesis_only_examples.sort(key=lambda x: x[1], reverse=True)
        
        biased_examples["hypothesis_only"] = [
            {
                "premise": ex[0]["premise"],
                "hypothesis": ex[0]["hypothesis"],
                "label": ex[0]["label"],
                "confidence": float(ex[1])
            }
            for ex in hypothesis_only_examples[:top_k]
        ]
        
        # Save biased examples
        output_path = self.output_dir / "most_biased_examples.json"
        with open(output_path, 'w') as f:
            json.dump(biased_examples, f, indent=2)
        
        return biased_examples