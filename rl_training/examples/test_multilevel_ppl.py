"""
Test script for multi-level perplexity calculation.

This demonstrates how to use the MultiLevelPerplexityCalculator
with sample text and a language model.

Usage:
    python test_multilevel_ppl.py --model_name "gpt2" --text "Your test text here."
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('..')

from multi_level_ppl import (
    MultiLevelPerplexityCalculator,
    compute_multi_level_reward,
    aggregate_multi_level_ppl
)


def prepare_inputs(text, tokenizer, model, max_length=512):
    """
    Prepare inputs for perplexity calculation.

    Args:
        text: Input text
        tokenizer: HuggingFace tokenizer
        model: Language model
        max_length: Maximum sequence length

    Returns:
        input_ids, attention_mask, labels
    """
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

    # For perplexity, labels are the same as input_ids
    # We'll mask padding tokens with -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels


def main():
    parser = argparse.ArgumentParser(description='Test multi-level perplexity calculation')
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='HuggingFace model name')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to evaluate (if not provided, uses example)')
    parser.add_argument('--baseline_text', type=str, default=None,
                        help='Baseline text for comparison')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Level Perplexity Test")
    print("=" * 60)

    # Use example texts if not provided
    if args.text is None:
        args.text = """
        The quick brown fox jumps over the lazy dog. This sentence contains
        every letter of the alphabet. It is often used for testing purposes.
        Language models should be able to predict this text well.
        """

    if args.baseline_text is None:
        args.baseline_text = """
        The rapid brown fox leaps over the sleepy dog. This sentence includes
        all letters of the alphabet. It is frequently used for testing reasons.
        Language models ought to predict this text accurately.
        """

    print(f"\nModel: {args.model_name}")
    print(f"Device: {args.device}")
    print("\n" + "-" * 60)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")

    # Initialize calculator
    calculator = MultiLevelPerplexityCalculator(
        tokenizer=tokenizer,
        phrase_sizes=[2, 3, 4],
        sentence_delimiters=r'[.!?。！？]'
    )

    # Test text perplexity
    print("\n" + "=" * 60)
    print("TEST TEXT EVALUATION")
    print("=" * 60)
    print(f"\nText:\n{args.text.strip()}\n")

    input_ids, attention_mask, labels = prepare_inputs(args.text, tokenizer, model)
    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)
    labels = labels.to(args.device)

    # Compute all levels
    with torch.no_grad():
        results = calculator.compute_all_levels(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            text=args.text
        )

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"Word-level PPL:     {results['word']['perplexity']:>10.4f}")
    print(f"Phrase-level PPL:   {results['phrase']['perplexity']:>10.4f}")
    print(f"Sentence-level PPL: {results['sentence']['perplexity']:>10.4f}")

    # Show phrase details if available
    if 'phrase_details' in results['phrase']['details']:
        print("\nPhrase-level breakdown:")
        for key, value in results['phrase']['details']['phrase_details'].items():
            print(f"  {key}: PPL = {value['perplexity']:.4f} ({value['num_chunks']} chunks)")

    # Aggregate
    agg_ppl = aggregate_multi_level_ppl(
        results['word']['perplexity'],
        results['phrase']['perplexity'],
        results['sentence']['perplexity'],
        weights=(0.2, 0.3, 0.5)
    )
    print(f"\nWeighted aggregate:  {agg_ppl:>10.4f}")

    current_ppls = {
        'word': results['word']['perplexity'],
        'phrase': results['phrase']['perplexity'],
        'sentence': results['sentence']['perplexity']
    }

    # Baseline text perplexity
    print("\n" + "=" * 60)
    print("BASELINE TEXT EVALUATION")
    print("=" * 60)
    print(f"\nText:\n{args.baseline_text.strip()}\n")

    baseline_ids, baseline_mask, baseline_labels = prepare_inputs(
        args.baseline_text, tokenizer, model
    )
    baseline_ids = baseline_ids.to(args.device)
    baseline_mask = baseline_mask.to(args.device)
    baseline_labels = baseline_labels.to(args.device)

    with torch.no_grad():
        baseline_results = calculator.compute_all_levels(
            model=model,
            input_ids=baseline_ids,
            attention_mask=baseline_mask,
            labels=baseline_labels,
            text=args.baseline_text
        )

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"Word-level PPL:     {baseline_results['word']['perplexity']:>10.4f}")
    print(f"Phrase-level PPL:   {baseline_results['phrase']['perplexity']:>10.4f}")
    print(f"Sentence-level PPL: {baseline_results['sentence']['perplexity']:>10.4f}")

    baseline_ppls = {
        'word': baseline_results['word']['perplexity'],
        'phrase': baseline_results['phrase']['perplexity'],
        'sentence': baseline_results['sentence']['perplexity']
    }

    # Compute reward
    print("\n" + "=" * 60)
    print("REWARD CALCULATION")
    print("=" * 60)

    reward, info = compute_multi_level_reward(
        baseline_ppls=baseline_ppls,
        current_ppls=current_ppls,
        weights=(0.2, 0.3, 0.5),
        reward_thresholds=[
            (0.05, 0.2),
            (0.5, 0.5),
            (1.0, 0.7),
            (2.0, 0.9),
            (3.0, 1.0),
        ]
    )

    print(f"\nComparison (Test vs Baseline):")
    print("-" * 60)
    print(f"Word improvement:     {info['improvements']['word']:>8.2f}%")
    print(f"Phrase improvement:   {info['improvements']['phrase']:>8.2f}%")
    print(f"Sentence improvement: {info['improvements']['sentence']:>8.2f}%")
    print(f"\nWeighted average:     {info['avg_improvement']:>8.2f}%")
    print(f"\nReward:               {reward:>8.4f}")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("-" * 60)
    if info['avg_improvement'] > 0:
        print(f"✓ Test text has {info['avg_improvement']:.2f}% better perplexity than baseline")
        print(f"✓ Reward score: {reward:.4f}")
    else:
        print(f"✗ Test text has {abs(info['avg_improvement']):.2f}% worse perplexity than baseline")
        print(f"✗ Reward score: {reward:.4f} (no improvement)")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
