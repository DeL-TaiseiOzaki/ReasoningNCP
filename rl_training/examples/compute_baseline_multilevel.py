"""
Compute multi-level baseline perplexities for a dataset.

This script computes word, phrase, and sentence-level perplexities
for each sample in a dataset using a baseline model. The results
are saved to a pickle file for use during training.

Usage:
    python compute_baseline_multilevel.py \
        --model_name "your-model" \
        --dataset_path "path/to/dataset" \
        --output_path "baseline_multilevel_ppls.pkl" \
        --num_samples 1000
"""

import argparse
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
import sys
sys.path.append('..')

from multi_level_ppl import MultiLevelPerplexityCalculator


def prepare_sample(sample, tokenizer, text_key='text', max_length=512):
    """
    Prepare a single sample for perplexity calculation.

    Args:
        sample: Dataset sample (dict)
        tokenizer: HuggingFace tokenizer
        text_key: Key for text in sample dict
        max_length: Maximum sequence length

    Returns:
        input_ids, attention_mask, labels, text
    """
    text = sample[text_key]

    # Tokenize
    encoded = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Labels for perplexity (mask padding)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels, text


def compute_dataset_baselines(
    model,
    tokenizer,
    dataset,
    calculator,
    text_key='text',
    id_key='id',
    batch_size=1,
    max_samples=None,
    device='cuda'
):
    """
    Compute multi-level baselines for entire dataset.

    Args:
        model: Language model
        tokenizer: Tokenizer
        dataset: HuggingFace dataset
        calculator: MultiLevelPerplexityCalculator instance
        text_key: Key for text in dataset
        id_key: Key for sample ID in dataset
        batch_size: Batch size (set to 1 for safety)
        max_samples: Maximum number of samples to process
        device: Device to use

    Returns:
        Dictionary mapping sample IDs to multi-level PPL baselines
    """
    model.eval()
    baselines = {}

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"\nProcessing {len(dataset)} samples...")

    for i, sample in enumerate(tqdm(dataset)):
        # Get sample ID
        sample_id = sample.get(id_key, i)

        try:
            # Prepare inputs
            input_ids, attention_mask, labels, text = prepare_sample(
                sample, tokenizer, text_key
            )

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Compute multi-level PPL
            with torch.no_grad():
                results = calculator.compute_all_levels(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    text=text
                )

            # Store results
            baselines[sample_id] = {
                'word': results['word']['perplexity'],
                'phrase': results['phrase']['perplexity'],
                'sentence': results['sentence']['perplexity'],
                'word_details': results['word']['details'],
                'phrase_details': results['phrase']['details'],
                'sentence_details': results['sentence']['details'],
                'text': text,  # Store text for reference
            }

            # Log progress every 100 samples
            if (i + 1) % 100 == 0:
                avg_word = sum(b['word'] for b in baselines.values()) / len(baselines)
                avg_phrase = sum(b['phrase'] for b in baselines.values()) / len(baselines)
                avg_sentence = sum(b['sentence'] for b in baselines.values()) / len(baselines)
                print(f"\n  Avg PPL so far - Word: {avg_word:.2f}, "
                      f"Phrase: {avg_phrase:.2f}, Sentence: {avg_sentence:.2f}")

        except Exception as e:
            print(f"\nError processing sample {sample_id}: {e}")
            continue

    return baselines


def main():
    parser = argparse.ArgumentParser(
        description='Compute multi-level baseline perplexities for a dataset'
    )
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset (HF dataset name or local path)')
    parser.add_argument('--dataset_split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--text_key', type=str, default='text',
                        help='Key for text in dataset')
    parser.add_argument('--id_key', type=str, default='id',
                        help='Key for sample ID in dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output pickle file path')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (None = all)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--phrase_sizes', type=int, nargs='+', default=[2, 3, 4],
                        help='N-gram sizes for phrase-level evaluation')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (currently only supports 1)')

    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Level Baseline Perplexity Computation")
    print("=" * 80)
    print(f"\nModel:           {args.model_name}")
    print(f"Dataset:         {args.dataset_path}")
    print(f"Split:           {args.dataset_split}")
    print(f"Output:          {args.output_path}")
    print(f"Max samples:     {args.num_samples if args.num_samples else 'All'}")
    print(f"Device:          {args.device}")
    print(f"Phrase sizes:    {args.phrase_sizes}")
    print("-" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32
    )
    model.to(args.device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    try:
        # Try loading as HuggingFace dataset
        dataset = load_dataset(args.dataset_path, split=args.dataset_split)
    except:
        # Try loading from disk
        try:
            dataset = load_from_disk(args.dataset_path)
            if args.dataset_split in dataset:
                dataset = dataset[args.dataset_split]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

    print(f"Dataset loaded: {len(dataset)} samples")

    # Initialize calculator
    calculator = MultiLevelPerplexityCalculator(
        tokenizer=tokenizer,
        phrase_sizes=args.phrase_sizes,
        sentence_delimiters=r'[.!?。！？]'
    )

    # Compute baselines
    print("\n" + "=" * 80)
    print("Computing baselines...")
    print("=" * 80)

    baselines = compute_dataset_baselines(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        calculator=calculator,
        text_key=args.text_key,
        id_key=args.id_key,
        batch_size=args.batch_size,
        max_samples=args.num_samples,
        device=args.device
    )

    # Compute statistics
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)

    word_ppls = [b['word'] for b in baselines.values()]
    phrase_ppls = [b['phrase'] for b in baselines.values()]
    sentence_ppls = [b['sentence'] for b in baselines.values()]

    print(f"\nSamples processed: {len(baselines)}")
    print("\nWord-level PPL:")
    print(f"  Mean:   {sum(word_ppls) / len(word_ppls):.4f}")
    print(f"  Min:    {min(word_ppls):.4f}")
    print(f"  Max:    {max(word_ppls):.4f}")
    print(f"  Median: {sorted(word_ppls)[len(word_ppls) // 2]:.4f}")

    print("\nPhrase-level PPL:")
    print(f"  Mean:   {sum(phrase_ppls) / len(phrase_ppls):.4f}")
    print(f"  Min:    {min(phrase_ppls):.4f}")
    print(f"  Max:    {max(phrase_ppls):.4f}")
    print(f"  Median: {sorted(phrase_ppls)[len(phrase_ppls) // 2]:.4f}")

    print("\nSentence-level PPL:")
    print(f"  Mean:   {sum(sentence_ppls) / len(sentence_ppls):.4f}")
    print(f"  Min:    {min(sentence_ppls):.4f}")
    print(f"  Max:    {max(sentence_ppls):.4f}")
    print(f"  Median: {sorted(sentence_ppls)[len(sentence_ppls) // 2]:.4f}")

    # Save baselines
    print(f"\nSaving baselines to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(baselines, f)

    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
