"""
Compute Multi-Level Baselines for TRL/Unsloth Training

This script computes word, phrase, and sentence-level baseline perplexities
for a dataset to be used with TRL GRPO training.

Usage:
    python compute_baselines_trl.py \
        --model_name gpt2 \
        --dataset_name your/dataset \
        --output_path baselines_trl.pkl \
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

from multi_level_ppl_trl import MultiLevelPPLCalculator

# Optional: Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False


def load_model(model_name, device='cuda'):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    if UNSLOTH_AVAILABLE:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            print("Loaded with Unsloth")
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            print("Loaded with transformers")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def prepare_text_from_sample(sample, text_key='text', prompt_key='prompt', completion_key='completion'):
    """
    Extract text from dataset sample.

    Supports various dataset formats:
    - Single text field
    - Prompt + completion
    - Other custom formats
    """
    if text_key in sample:
        return sample[text_key]
    elif prompt_key in sample and completion_key in sample:
        return sample[prompt_key] + sample[completion_key]
    elif prompt_key in sample:
        return sample[prompt_key]
    elif 'input' in sample:
        return sample['input']
    else:
        # Try to concatenate all string fields
        text_parts = [str(v) for v in sample.values() if isinstance(v, str)]
        return ' '.join(text_parts) if text_parts else ''


def compute_baselines(
    model,
    tokenizer,
    dataset,
    calculator,
    text_key='text',
    id_key='id',
    max_samples=None,
    batch_size=4,
):
    """
    Compute multi-level baselines for dataset.

    Args:
        model: Language model
        tokenizer: Tokenizer
        dataset: HuggingFace dataset
        calculator: MultiLevelPPLCalculator
        text_key: Key for text in dataset
        id_key: Key for sample ID
        max_samples: Max samples to process
        batch_size: Batch size

    Returns:
        Dictionary mapping IDs/texts to multi-level PPLs
    """
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"\nProcessing {len(dataset)} samples in batches of {batch_size}...")

    baselines = {}
    errors = 0

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]

        # Extract texts
        if isinstance(batch, dict):
            # Single sample
            texts = [prepare_text_from_sample(batch, text_key)]
            ids = [batch.get(id_key, i)]
        else:
            # Multiple samples
            texts = [prepare_text_from_sample(sample, text_key) for sample in batch]
            ids = [sample.get(id_key, i + j) for j, sample in enumerate(batch)]

        try:
            # Compute PPLs
            with torch.no_grad():
                ppls = calculator.compute_all_levels(
                    model=model,
                    texts=texts
                )

            # Store results
            for j, (text, sample_id) in enumerate(zip(texts, ids)):
                # Handle both single and batch results
                if isinstance(ppls['word'], torch.Tensor) and ppls['word'].numel() > 1:
                    word_ppl = ppls['word'][j].item()
                    phrase_ppl = ppls['phrase'][j].item()
                    sentence_ppl = ppls['sentence'][j].item()
                elif isinstance(ppls['word'], torch.Tensor):
                    word_ppl = ppls['word'].item()
                    phrase_ppl = ppls['phrase'].item()
                    sentence_ppl = ppls['sentence'].item()
                else:
                    word_ppl = ppls['word']
                    phrase_ppl = ppls['phrase']
                    sentence_ppl = ppls['sentence']

                baselines[str(sample_id)] = {
                    'word': word_ppl,
                    'phrase': phrase_ppl,
                    'sentence': sentence_ppl,
                    'text': text[:200],  # Store snippet for reference
                }

        except Exception as e:
            print(f"\nError processing batch at index {i}: {e}")
            errors += 1
            continue

        # Log progress every 100 samples
        if (i + batch_size) % 100 == 0 and len(baselines) > 0:
            avg_word = sum(b['word'] for b in baselines.values()) / len(baselines)
            avg_phrase = sum(b['phrase'] for b in baselines.values()) / len(baselines)
            avg_sentence = sum(b['sentence'] for b in baselines.values()) / len(baselines)
            tqdm.write(f"  Progress: {len(baselines)} samples | "
                      f"Avg PPL - Word: {avg_word:.2f}, Phrase: {avg_phrase:.2f}, Sentence: {avg_sentence:.2f}")

    print(f"\nCompleted with {errors} errors")
    return baselines


def main():
    parser = argparse.ArgumentParser(description='Compute multi-level baselines for TRL')

    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name or path')
    parser.add_argument('--dataset_split', type=str, default='train',
                       help='Dataset split')
    parser.add_argument('--text_key', type=str, default='text',
                       help='Key for text in dataset')
    parser.add_argument('--id_key', type=str, default='id',
                       help='Key for sample ID')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output pickle file path')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')
    parser.add_argument('--phrase_sizes', type=int, nargs='+', default=[2, 3, 4],
                       help='N-gram sizes for phrase-level')

    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Level Baseline Computation (TRL)")
    print("=" * 80)
    print(f"Model:        {args.model_name}")
    print(f"Dataset:      {args.dataset_name}")
    print(f"Split:        {args.dataset_split}")
    print(f"Output:       {args.output_path}")
    print(f"Max samples:  {args.num_samples if args.num_samples else 'All'}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Device:       {args.device}")
    print("-" * 80)

    # Load model
    model, tokenizer = load_model(args.model_name, args.device)

    # Load dataset
    print(f"\nLoading dataset...")
    try:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    except:
        try:
            dataset = load_from_disk(args.dataset_name)
            if args.dataset_split in dataset:
                dataset = dataset[args.dataset_split]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

    print(f"Dataset loaded: {len(dataset)} samples")

    # Initialize calculator
    calculator = MultiLevelPPLCalculator(
        tokenizer=tokenizer,
        phrase_sizes=args.phrase_sizes,
        device=args.device
    )

    # Compute baselines
    print("\n" + "=" * 80)
    print("Computing baselines...")
    print("=" * 80)

    baselines = compute_baselines(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        calculator=calculator,
        text_key=args.text_key,
        id_key=args.id_key,
        max_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    # Statistics
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

    # Save
    print(f"\nSaving baselines to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(baselines, f)

    print("\nDone!")
    print("=" * 80)


if __name__ == '__main__':
    main()
