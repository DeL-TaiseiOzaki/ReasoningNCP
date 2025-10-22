"""
GRPO Training with Multi-Level PPL Reward Function using Unsloth + TRL

This script demonstrates how to train a language model using GRPO (Group Relative Policy Optimization)
with multi-level perplexity rewards, using Unsloth for fast training and TRL for RL.

Requirements:
    pip install trl transformers unsloth accelerate wandb

Usage:
    python train_grpo_multilevel_unsloth.py \
        --model_name unsloth/llama-3-8b-bnb-4bit \
        --dataset_name your/dataset \
        --baselines_path baselines.pkl \
        --output_dir ./output
"""

import argparse
import torch
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import sys
sys.path.append('..')

from multi_level_ppl_trl import MultiLevelPPLRewardFunction

# Optional: Unsloth for fast training
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    from transformers import AutoModelForCausalLM
    UNSLOTH_AVAILABLE = False
    print("Unsloth not available, using standard transformers")


def load_model_and_tokenizer(model_name, max_seq_length=2048, load_in_4bit=True):
    """
    Load model and tokenizer with optional Unsloth acceleration.

    Args:
        model_name: HuggingFace model name
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to use 4-bit quantization

    Returns:
        model, tokenizer
    """
    if UNSLOTH_AVAILABLE:
        print("Loading model with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,
        )

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        print("Loading model with standard transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_dataset(dataset_name, tokenizer, split='train', max_samples=None):
    """
    Prepare dataset for GRPO training.

    Args:
        dataset_name: HuggingFace dataset name or path
        tokenizer: Tokenizer
        split: Dataset split
        max_samples: Maximum number of samples

    Returns:
        Dataset with 'query' column
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # GRPO expects a 'query' column with prompts
    # Modify this based on your dataset structure
    def format_prompt(example):
        # Adjust this to match your dataset format
        if 'prompt' in example:
            example['query'] = example['prompt']
        elif 'text' in example:
            # For datasets with full text, extract prompt somehow
            example['query'] = example['text'][:100] + "..."
        elif 'input' in example:
            example['query'] = example['input']
        else:
            raise ValueError("Dataset must have 'prompt', 'text', or 'input' column")

        return example

    dataset = dataset.map(format_prompt)

    print(f"Dataset loaded: {len(dataset)} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description='GRPO training with multi-level PPL rewards')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='unsloth/llama-3-8b-bnb-4bit',
                       help='Model name or path')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                       help='Use 4-bit quantization')

    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name or path')
    parser.add_argument('--dataset_split', type=str, default='train',
                       help='Dataset split')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to use')

    # Reward function arguments
    parser.add_argument('--baselines_path', type=str, default=None,
                       help='Path to baselines pickle file')
    parser.add_argument('--ppl_weights', type=float, nargs=3, default=[0.2, 0.3, 0.5],
                       help='Weights for word, phrase, sentence PPL')
    parser.add_argument('--normalize_rewards', action='store_true',
                       help='Normalize rewards')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./output_grpo_multilevel',
                       help='Output directory')
    parser.add_argument('--num_train_epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')

    # Logging
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=100,
                       help='Save checkpoint frequency')
    parser.add_argument('--wandb_project', type=str, default='grpo-multilevel-ppl',
                       help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for logging')

    args = parser.parse_args()

    print("=" * 80)
    print("GRPO Training with Multi-Level PPL Rewards")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Baselines: {args.baselines_path}")
    print(f"PPL Weights: word={args.ppl_weights[0]}, phrase={args.ppl_weights[1]}, sentence={args.ppl_weights[2]}")
    print(f"Output: {args.output_dir}")
    print("-" * 80)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit
    )

    # Prepare dataset
    dataset = prepare_dataset(
        args.dataset_name,
        tokenizer,
        split=args.dataset_split,
        max_samples=args.max_samples
    )

    # Create reward function
    print("\nInitializing multi-level PPL reward function...")
    reward_fn = MultiLevelPPLRewardFunction(
        baseline_model=model,
        tokenizer=tokenizer,
        baselines=args.baselines_path,
        weights=tuple(args.ppl_weights),
        normalize=args.normalize_rewards,
    )
    print("Reward function initialized!")

    # Configure GRPO training
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name or f"grpo-multilevel-{args.model_name.split('/')[-1]}",
    )

    # Initialize GRPO trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        # Generation config
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("Trainer initialized!")
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    # Train
    trainer.train()

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Save final model
    print(f"\nSaving model to {args.output_dir}/final...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    print("\nDone!")


if __name__ == '__main__':
    main()
