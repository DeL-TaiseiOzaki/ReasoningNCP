"""
Test Multi-Level PPL Reward Function for TRL

Quick test to verify the TRL-compatible reward function works correctly.

Usage:
    python test_multilevel_ppl_trl.py --model_name gpt2
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('..')

from multi_level_ppl_trl import MultiLevelPPLRewardFunction, MultiLevelPPLCalculator


def main():
    parser = argparse.ArgumentParser(description='Test TRL multi-level PPL reward function')
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Model name')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')
    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Level PPL Reward Function Test (TRL)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print("-" * 80)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        device_map='auto'
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!")

    # Test data
    prompts = [
        "The quick brown fox",
        "In a distant galaxy",
        "Once upon a time",
    ]

    completions = [
        " jumps over the lazy dog. This is a well-known pangram.",
        " far far away, there lived a civilization of advanced beings.",
        " there was a little girl who loved to read books every day.",
    ]

    print("\n" + "=" * 80)
    print("Testing Multi-Level PPL Calculator")
    print("=" * 80)

    calculator = MultiLevelPPLCalculator(
        tokenizer=tokenizer,
        device=args.device
    )

    # Test single text
    test_text = prompts[0] + completions[0]
    print(f"\nTest text: {test_text}")

    ppls = calculator.compute_all_levels(model=model, texts=test_text)

    print("\nResults:")
    print(f"  Word-level PPL:     {ppls['word']:.4f}")
    print(f"  Phrase-level PPL:   {ppls['phrase']:.4f}")
    print(f"  Sentence-level PPL: {ppls['sentence']:.4f}")

    # Test batch
    print("\n" + "-" * 80)
    print("Testing batch processing...")

    full_texts = [p + c for p, c in zip(prompts, completions)]
    batch_ppls = calculator.compute_all_levels(model=model, texts=full_texts)

    print("\nBatch results:")
    for i, text in enumerate(full_texts):
        if isinstance(batch_ppls['word'], torch.Tensor):
            word = batch_ppls['word'][i].item()
            phrase = batch_ppls['phrase'][i].item()
            sentence = batch_ppls['sentence'][i].item()
        else:
            word = batch_ppls['word']
            phrase = batch_ppls['phrase']
            sentence = batch_ppls['sentence']

        print(f"\nText {i+1}: {text[:50]}...")
        print(f"  Word: {word:.4f}, Phrase: {phrase:.4f}, Sentence: {sentence:.4f}")

    # Test reward function
    print("\n" + "=" * 80)
    print("Testing Reward Function")
    print("=" * 80)

    # Create reward function without baselines
    print("\nInitializing reward function...")
    reward_fn = MultiLevelPPLRewardFunction(
        baseline_model=model,
        tokenizer=tokenizer,
        baselines=None,  # No baselines - will use current as baseline
        weights=(0.2, 0.3, 0.5),
        device=args.device
    )

    # Compute rewards
    print("\nComputing rewards...")
    rewards = reward_fn(prompts=prompts, completions=completions)

    print("\nRewards (without baselines, all should be ~0):")
    for i, (prompt, completion, reward) in enumerate(zip(prompts, completions, rewards)):
        print(f"\n{i+1}. Prompt: {prompt}")
        print(f"   Completion: {completion}")
        print(f"   Reward: {reward:.4f}")

    # Test with mock baselines
    print("\n" + "-" * 80)
    print("Testing with mock baselines...")

    mock_baselines = {
        prompts[0]: {'word': 50.0, 'phrase': 55.0, 'sentence': 60.0},
        prompts[1]: {'word': 45.0, 'phrase': 50.0, 'sentence': 55.0},
        prompts[2]: {'word': 40.0, 'phrase': 44.0, 'sentence': 48.0},
    }

    reward_fn_with_baselines = MultiLevelPPLRewardFunction(
        baseline_model=model,
        tokenizer=tokenizer,
        baselines=mock_baselines,
        weights=(0.2, 0.3, 0.5),
        device=args.device
    )

    rewards, details = reward_fn_with_baselines.compute_rewards(
        texts=full_texts,
        prompts=prompts,
        completions=completions,
        return_details=True
    )

    print("\nRewards (with mock baselines):")
    for i, (prompt, reward) in enumerate(zip(prompts, rewards)):
        improvements = details['improvements']
        word_imp = improvements['word'][i].item()
        phrase_imp = improvements['phrase'][i].item()
        sentence_imp = improvements['sentence'][i].item()
        avg_imp = improvements['avg'][i].item()

        print(f"\n{i+1}. Prompt: {prompt}")
        print(f"   Improvements: Word={word_imp:.2f}%, Phrase={phrase_imp:.2f}%, Sentence={sentence_imp:.2f}%")
        print(f"   Average improvement: {avg_imp:.2f}%")
        print(f"   Reward: {reward:.4f}")

    # Test callable interface (for TRL)
    print("\n" + "=" * 80)
    print("Testing TRL Callable Interface")
    print("=" * 80)

    print("\nCalling reward_fn(prompts, completions)...")
    trl_rewards = reward_fn_with_baselines(prompts=prompts, completions=completions)

    print("\nTRL-style rewards:")
    print(f"  Type: {type(trl_rewards)}")
    print(f"  Shape: {trl_rewards.shape}")
    print(f"  Values: {trl_rewards.tolist()}")

    # Verify it's a tensor
    assert isinstance(trl_rewards, torch.Tensor), "Rewards must be a tensor"
    assert trl_rewards.shape[0] == len(prompts), "Batch size mismatch"

    print("\n✓ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
