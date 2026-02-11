import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser, \
        EarlyStoppingCallback
import evaluate
from pathlib import Path
# import helpers
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import torch
import numpy as np
from cartography import CartographyCallback
from bias_analysis import BiasAnalyzer
# Add to imports
from eval_utils import DetailedEvaluator
from visualization import ResultsVisualizer


NUM_PREPROCESSING_WORKERS = 2

class CartographyTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to feed data to CartographyCallback"""
        outputs = super().training_step(model, inputs, num_items_in_batch)
        
        # Get logits from model output
        _, outputs_dict = self.compute_loss(model, inputs, return_outputs=True)
        
        # Convert scalar loss to list and ensure logits are in list format
        model.last_batch_metrics = {
            'losses': [outputs.detach().cpu().numpy().item()],  # Convert scalar to list
            'logits': outputs_dict.logits.detach().cpu().numpy().tolist(),
            'labels': inputs['labels'].detach().cpu().numpy().tolist()
        }
        return outputs

def json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj

# Compute_metrics function to track more detailed metrics
def compute_detailed_metrics(eval_preds):
    """Compute detailed evaluation metrics ensuring all values are JSON serializable."""
    if eval_preds is None:
        return {}
        
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    
    # Get predicted labels
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate basic metrics
    accuracy = float(np.mean(predicted_labels == labels))
    
    # Calculate per-class metrics
    metrics = {
        "accuracy": accuracy,
        "per_class_accuracy": {},
        "confusion_matrix": None
    }
    
    # Per-class accuracy
    for label in range(3):
        label_mask = labels == label
        if label_mask.sum() > 0:
            label_accuracy = float((predicted_labels[label_mask] == labels[label_mask]).mean())
            metrics["per_class_accuracy"][f"class_{label}"] = label_accuracy
    
    # Confusion matrix
    conf_matrix = np.zeros((3, 3))
    for i in range(len(labels)):
        conf_matrix[labels[i]][predicted_labels[i]] += 1
    metrics["confusion_matrix"] = conf_matrix.tolist()  # Convert to list
    
    return metrics


# This function wraps the compute_detailed_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
def compute_metrics_and_store_predictions(eval_preds):
    global eval_predictions
    eval_predictions = eval_preds
    return compute_detailed_metrics(eval_preds)


def main():
    global eval_predictions
    eval_predictions = None

    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    # Add new arguments for our analysis
    argp.add_argument('--enable_cartography', action='store_true',
                      help='Enable dataset cartography analysis')
    argp.add_argument('--early_stopping_patience', type=int, default=3,
                      help='Number of epochs to wait for improvement before early stopping')
    
    # Add new argument for bias analysis
    argp.add_argument('--analyze_bias', action='store_true',
                    help='Run bias analysis on the evaluation dataset')

    # Add new argument
    argp.add_argument('--create_visualizations', action='store_true',
                    help='Create visualizations of evaluation results')

    training_args, args = argp.parse_args_into_dataclasses()

    # Modify training arguments to enable checkpoint saving
    if training_args.do_train:
        training_args.save_strategy = "steps"
        training_args.save_steps = 500
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = 500
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "accuracy"
        training_args.greater_is_better = True
        
        # Enable gradient checkpointing for memory efficiency
        training_args.gradient_checkpointing = True

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_metrics_and_store_predictions

    # Add cartography trainer if enabled
    if args.enable_cartography:
        trainer_class = CartographyTrainer
    

    # Initialize callbacks list
    callbacks = []
    
    # Add early stopping
    if training_args.do_train:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))
    
    # Add cartography callback if enabled
    if args.enable_cartography:
        cartography_callback = CartographyCallback(
            output_dir=Path(training_args.output_dir) / "cartography"
        )
        callbacks.append(cartography_callback)


    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
        callbacks=callbacks
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

        if args.enable_cartography:
            print("Computing cartography metrics...")
            cartography_metrics = cartography_callback.compute_cartography_metrics()
            
            # Save cartography metrics summary
            metrics_path = Path(training_args.output_dir) / "cartography_summary.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    'avg_confidence': np.mean(cartography_metrics['confidence']),
                    'avg_variability': np.mean(cartography_metrics['variability']),
                    'avg_correctness': np.mean(cartography_metrics['correctness']),
                }, f, indent=2)

        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # Add error analysis if requested
        if eval_predictions is None or not hasattr(eval_predictions, 'predictions'):
            print("Warning: No predictions available for error analysis")
        else:
            if args.task == 'nli':
                print("\nRunning error analysis...")    
            error_analysis = {
                'most_confused_pairs': [],
                'systematic_errors': {}
            }
        
            # Find most confused premise-hypothesis pairs
            for i, example in enumerate(eval_dataset):
                predicted_label = int(eval_predictions.predictions[i].argmax())
                if predicted_label != example['label']:
                    error_analysis['most_confused_pairs'].append({
                        'premise': example['premise'],
                        'hypothesis': example['hypothesis'],
                        'true_label': example['label'],
                        'predicted_label': predicted_label,
                        'confidence': float(eval_predictions.predictions[i].max())
                    })
        
            # Sort by confidence to find high-confidence errors
            error_analysis['most_confused_pairs'].sort(key=lambda x: x['confidence'], reverse=True)
            
            # Save error analysis
            with open(os.path.join(training_args.output_dir, 'error_analysis.json'), 'w') as f:
                json.dump(error_analysis, f, indent=2)

        # Add to evaluation section
                  
        if args.analyze_bias and args.task == 'nli':
            print("\nRunning bias analysis...")
            bias_analyzer = BiasAnalyzer(
                model_path=args.model,
                output_dir=os.path.join(training_args.output_dir, 'bias_analysis')
            )
            
            # Run full bias analysis
            bias_results = bias_analyzer.run_full_analysis(eval_dataset)
            
            # Get most biased examples
            biased_examples = bias_analyzer.get_most_biased_examples(eval_dataset)
            
            # Add bias metrics to overall results
            results.update({
                "bias_analysis": bias_results
            })


        # Create detailed evaluator
        detailed_evaluator = DetailedEvaluator(
            output_dir=os.path.join(training_args.output_dir, 'detailed_eval')
        )
        
        if eval_predictions is not None and hasattr(eval_predictions, 'predictions'):
            # Get detailed metrics
            detailed_metrics = detailed_evaluator.compute_metrics(
                examples=eval_dataset,
                logits=eval_predictions.predictions,
                labels=eval_predictions.label_ids
            )
            
            # Add detailed metrics to results
            results.update({
                "detailed_metrics": detailed_metrics.__dict__
            })
        else:
            print("Warning: Skipping detailed evaluation due to missing predictions")
        
        # Create visualizations if requested
        if args.create_visualizations:
            visualizer = ResultsVisualizer(
                output_dir=os.path.join(training_args.output_dir, 'visualizations')
            )
            
            # Create all visualizations
            visualizer.create_all_visualizations(
                eval_metrics_path=os.path.join(
                    training_args.output_dir,
                    'detailed_eval/detailed_metrics.json'
                ),
                bias_results_path=os.path.join(
                    training_args.output_dir,
                    'bias_analysis/bias_analysis_results.json'
                ) if args.analyze_bias else None
            )

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        # Convert numpy types to native Python types for JSON serialization
        # results = {k: json_serializable(v) for k, v in results.items()}

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, 
                                np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        # Convert all results to JSON serializable format
        results = convert_to_serializable(results)


        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f, indent=2)

        # Save predictions
        if eval_predictions is not None and hasattr(eval_predictions, 'predictions'):
            with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
                if args.task == 'qa':
                    predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                    for example in eval_dataset:
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')
                else:
                    for i, example in enumerate(eval_dataset):
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')
        else:
            print("Warning: No predictions available to save")


if __name__ == "__main__":
    main()
