import torch
from transformers import TrainerCallback
import numpy as np
import json
from pathlib import Path

class CartographyCallback(TrainerCallback):
    """Callback for collecting training dynamics."""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "cartography"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_dynamics = {
            'example_ids': [],
            'confidences': [],
            'variabilities': [],
            'losses': []
        }
        self.current_epoch = 0
        self.epoch_metrics = {}

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = state.epoch
        self.epoch_metrics = {
            'losses': [],
            'logits': [],
            'predictions': []
        }

    def on_step_end(self, args, state, control, model, **kwargs):
        """Collect logits and losses for each step."""
        if hasattr(model, 'module'):
            model = model.module

        # Get last batch metrics from model
        if hasattr(model, 'last_batch_metrics'):
            batch_metrics = model.last_batch_metrics
            if batch_metrics is not None and 'losses' in batch_metrics:
                # Extend lists with batch data
                self.epoch_metrics['losses'].extend(batch_metrics['losses'])
                self.epoch_metrics['logits'].extend(batch_metrics['logits'])

    def on_epoch_end(self, args, state, control, **kwargs):
        """Calculate variability and confidence at epoch end."""
        # Skip if no metrics collected
        if not self.epoch_metrics['losses'] or not self.epoch_metrics['logits']:
            return
            
        # Convert lists to numpy arrays for efficient computation
        losses = np.array(self.epoch_metrics['losses'])
        logits = np.array(self.epoch_metrics['logits'])
        
        # Calculate confidence (probability of predicted class)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        if probs.shape[0] > 0:  # Check if we have any probabilities
            confidences = probs.max(dim=-1).values.numpy()
            
            # Save epoch metrics
            epoch_data = {
                'epoch': self.current_epoch,
                'losses': losses.tolist(),
                'confidences': confidences.tolist()
            }
            
            # Save to disk to manage memory
            save_path = self.output_dir / f'epoch_{self.current_epoch}_dynamics.json'
            with open(save_path, 'w') as f:
                json.dump(epoch_data, f)

    def compute_cartography_metrics(self):
        """Compute final cartography metrics after training."""
        # Load all epoch data
        all_epochs = sorted(self.output_dir.glob('epoch_*_dynamics.json'))
        
        if not all_epochs:  # Check if we have any epoch data
            return {
                'confidence': [],
                'variability': [],
                'correctness': []
            }
        
        total_epochs = len(all_epochs)
        example_metrics = {}
        
        for epoch_file in all_epochs:
            with open(epoch_file, 'r') as f:
                epoch_data = json.load(f)
                
            for idx, (conf, loss) in enumerate(zip(epoch_data.get('confidences', []), 
                                                 epoch_data.get('losses', []))):
                if idx not in example_metrics:
                    example_metrics[idx] = {
                        'confidences': [],
                        'losses': []
                    }
                example_metrics[idx]['confidences'].append(conf)
                example_metrics[idx]['losses'].append(loss)
        
        # Compute final metrics
        final_metrics = {
            'variability': [],
            'confidence': [],
            'correctness': []
        }
        
        for idx, metrics in example_metrics.items():
            confs = np.array(metrics['confidences'])
            final_metrics['variability'].append(float(np.std(confs)))
            final_metrics['confidence'].append(float(np.mean(confs)))
            final_metrics['correctness'].append(
                float(np.mean(np.array(metrics['losses']) < 0.5))
            )
        
        # Save final metrics
        with open(self.output_dir / 'cartography_metrics.json', 'w') as f:
            json.dump(final_metrics, f)
        
        return final_metrics