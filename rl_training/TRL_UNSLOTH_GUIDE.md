# Multi-Level PPL for TRL and Unsloth

Complete guide for using multi-level perplexity rewards with TRL's GRPO trainer and Unsloth for fast training.

## Overview

This implementation provides a **TRL-compatible** multi-level perplexity reward function that works seamlessly with:
- **TRL (Transformers Reinforcement Learning)**: State-of-the-art RL library
- **GRPO (Group Relative Policy Optimization)**: Efficient RL algorithm
- **Unsloth**: 2-5x faster training with lower memory usage
- **Standard HuggingFace models**: Works with any causal LM

### Why TRL + Unsloth?

| Feature | Ray + OpenRLHF | TRL + Unsloth |
|---------|----------------|---------------|
| **Setup complexity** | High (distributed) | Low (single node) |
| **Training speed** | Fast (distributed) | Very fast (optimized) |
| **Memory usage** | High | Low (4-bit quantization) |
| **Code simplicity** | Complex | Simple |
| **Best for** | Large-scale clusters | Single GPU / small setups |

## Installation

```bash
# Core dependencies
pip install torch transformers accelerate

# TRL for GRPO
pip install trl

# Unsloth for fast training (optional but recommended)
pip install unsloth

# For logging
pip install wandb
```

### Unsloth Installation (Recommended)

```bash
# For CUDA 12.1
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8
pip install "unsloth[cu118-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# For other versions, see: https://github.com/unslothai/unsloth
```

## Quick Start

### 1. Test the Reward Function

First, verify everything works:

```bash
cd rl_training/examples

# Basic test
python test_multilevel_ppl_trl.py --model_name gpt2

# With a larger model
python test_multilevel_ppl_trl.py --model_name meta-llama/Llama-2-7b-hf
```

Expected output:
```
Multi-Level PPL Reward Function Test (TRL)
============================================================
Model: gpt2
Device: cuda

Loading model...
Model loaded!

Testing Multi-Level PPL Calculator
============================================================

Test text: The quick brown fox jumps over the lazy dog...

Results:
  Word-level PPL:     45.2341
  Phrase-level PPL:   47.8932
  Sentence-level PPL: 51.2104

✓ All tests passed!
```

### 2. Compute Baselines

Compute multi-level baselines for your dataset:

```bash
python compute_baselines_trl.py \
    --model_name gpt2 \
    --dataset_name wikitext \
    --dataset_split test \
    --output_path baselines_wiki.pkl \
    --num_samples 1000 \
    --batch_size 4
```

### 3. Train with GRPO

Run GRPO training with multi-level PPL rewards:

```bash
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --dataset_name your/dataset \
    --baselines_path baselines_wiki.pkl \
    --output_dir ./output_grpo \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5
```

## Detailed Usage

### Creating a Reward Function

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from multi_level_ppl_trl import MultiLevelPPLRewardFunction

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create reward function
reward_fn = MultiLevelPPLRewardFunction(
    baseline_model=model,
    tokenizer=tokenizer,
    baselines="baselines.pkl",  # Optional: pre-computed baselines
    weights=(0.2, 0.3, 0.5),    # (word, phrase, sentence)
    reward_thresholds=[
        (0.05, 0.2),  # >= 0.05% improvement -> 0.2 reward
        (0.5, 0.5),   # >= 0.5% improvement -> 0.5 reward
        (1.0, 0.7),   # >= 1% improvement -> 0.7 reward
        (2.0, 0.9),   # >= 2% improvement -> 0.9 reward
        (3.0, 1.0),   # >= 3% improvement -> 1.0 reward
    ],
    normalize=False,  # Whether to normalize rewards
    device='cuda'
)
```

### Using with TRL GRPOTrainer

```python
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("your/dataset", split="train")

# Configure GRPO
config = GRPOConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
)

# Create trainer
trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_fn=reward_fn,  # ← Our multi-level PPL reward function
    max_new_tokens=128,
    temperature=0.7,
)

# Train!
trainer.train()
```

### Using with Unsloth for Fast Training

```python
from unsloth import FastLanguageModel
from multi_level_ppl_trl import MultiLevelPPLRewardFunction
from trl import GRPOConfig, GRPOTrainer

# Load model with Unsloth (2-5x faster!)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
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
)

# Create reward function (same as before)
reward_fn = MultiLevelPPLRewardFunction(
    baseline_model=model,
    tokenizer=tokenizer,
    baselines="baselines.pkl",
    weights=(0.2, 0.3, 0.5),
)

# Create trainer and train
trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_fn=reward_fn,
)

trainer.train()
```

## Dataset Format

Your dataset should have a `query` column with prompts. If not, the training script will try to auto-format:

```python
# Example dataset formats

# Format 1: Prompt column
{
    "prompt": "The quick brown fox",
    "id": "sample_001"
}

# Format 2: Text column
{
    "text": "Complete this story: Once upon a time...",
    "id": "sample_002"
}

# Format 3: Input column
{
    "input": "Write a poem about nature:",
    "id": "sample_003"
}
```

To customize formatting, edit `prepare_dataset()` in `train_grpo_multilevel_unsloth.py`.

## Configuration Options

### Reward Function Parameters

```python
reward_fn = MultiLevelPPLRewardFunction(
    baseline_model=model,        # Model for PPL computation
    tokenizer=tokenizer,          # Tokenizer
    baselines="baselines.pkl",    # Pre-computed baselines (optional)
    weights=(0.2, 0.3, 0.5),     # Importance of word/phrase/sentence
    reward_thresholds=[...],      # Improvement % -> reward mapping
    normalize=False,              # Normalize rewards (mean=0, std=1)
    device='cuda',                # Device
    use_cache=True,               # Cache computed PPLs
)
```

### Weight Configurations

Different tasks benefit from different weights:

```python
# Story generation (emphasis on sentence coherence)
weights = (0.1, 0.3, 0.6)

# Technical writing (emphasis on word precision)
weights = (0.4, 0.4, 0.2)

# Balanced
weights = (0.33, 0.33, 0.34)

# Code generation (emphasis on local correctness)
weights = (0.4, 0.5, 0.1)
```

### Reward Thresholds

Customize how improvements map to rewards:

```python
# More granular rewards
reward_thresholds = [
    (0.01, 0.1),   # Any tiny improvement
    (0.1, 0.3),
    (0.5, 0.5),
    (1.0, 0.7),
    (2.0, 0.9),
    (5.0, 1.0),
]

# Stricter rewards (only reward significant improvements)
reward_thresholds = [
    (1.0, 0.3),
    (2.0, 0.6),
    (5.0, 1.0),
]

# Linear mapping
reward_thresholds = [
    (0.0, 0.0),
    (1.0, 0.25),
    (2.0, 0.5),
    (3.0, 0.75),
    (4.0, 1.0),
]
```

### Training Parameters

```python
# Fast prototyping
config = GRPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
)

# Production training
config = GRPOConfig(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=100,
    lr_scheduler_type="cosine",
)

# Memory-constrained
config = GRPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
)
```

## Examples

### Example 1: Training on Custom Dataset

```bash
# 1. Prepare your dataset in HuggingFace format
# dataset/
#   ├── train.json
#   └── test.json

# 2. Compute baselines
python compute_baselines_trl.py \
    --model_name gpt2 \
    --dataset_name ./dataset \
    --dataset_split train \
    --output_path baselines_custom.pkl \
    --num_samples 5000

# 3. Train with GRPO
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/mistral-7b-bnb-4bit \
    --dataset_name ./dataset \
    --baselines_path baselines_custom.pkl \
    --output_dir ./output_mistral \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --wandb_project my-grpo-project
```

### Example 2: Fine-tuning Llama 3 with Unsloth

```bash
# Compute baselines with Llama 3
python compute_baselines_trl.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --dataset_name your/dataset \
    --output_path baselines_llama3.pkl \
    --num_samples 10000 \
    --batch_size 8

# Train with custom weights (emphasize sentence coherence)
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --dataset_name your/dataset \
    --baselines_path baselines_llama3.pkl \
    --ppl_weights 0.1 0.3 0.6 \
    --output_dir ./output_llama3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --max_new_tokens 256 \
    --temperature 0.8
```

### Example 3: Without Baselines (Zero-Shot)

You can train without pre-computed baselines:

```python
# No baselines - current PPL becomes the baseline
reward_fn = MultiLevelPPLRewardFunction(
    baseline_model=model,
    tokenizer=tokenizer,
    baselines=None,  # No baselines
    weights=(0.2, 0.3, 0.5),
)

# This will give 0 rewards initially, but rewards will emerge
# as the model improves relative to its initial state
```

## Performance Tips

### Memory Optimization

```python
# Use 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    load_in_4bit=True,  # ← Reduces memory by 4x
)

# Smaller batch size with more gradient accumulation
config = GRPOConfig(
    per_device_train_batch_size=1,  # Smaller
    gradient_accumulation_steps=16,  # More steps
)

# Enable gradient checkpointing
config = GRPOConfig(
    gradient_checkpointing=True,
)
```

### Speed Optimization

```python
# Use Unsloth (2-5x faster)
from unsloth import FastLanguageModel

# Use bfloat16
config = GRPOConfig(
    bf16=True,  # Faster than fp16 on modern GPUs
)

# Batch baseline computation
calculator = MultiLevelPPLCalculator(...)
ppls = calculator.compute_all_levels(
    model=model,
    texts=batch_of_texts,  # Process multiple texts at once
)
```

### Training Stability

```python
# Normalize rewards
reward_fn = MultiLevelPPLRewardFunction(
    ...,
    normalize=True,  # ← Helps with training stability
)

# Use warmup
config = GRPOConfig(
    warmup_steps=100,
    lr_scheduler_type="cosine",
)

# Clip gradients
config = GRPOConfig(
    max_grad_norm=1.0,
)
```

## Monitoring Training

### WandB Integration

```bash
# Enable WandB logging
python train_grpo_multilevel_unsloth.py \
    --wandb_project my-project \
    --run_name llama3-multilevel-ppl \
    ...
```

Metrics logged:
- `train/reward`: Overall reward
- `train/reward_word`: Word-level improvement
- `train/reward_phrase`: Phrase-level improvement
- `train/reward_sentence`: Sentence-level improvement
- `train/loss`: Training loss
- `train/learning_rate`: Current LR

### Custom Logging

```python
# In your training script
reward_fn = MultiLevelPPLRewardFunction(...)

# Get detailed metrics
rewards, details = reward_fn.compute_rewards(
    texts=texts,
    return_details=True
)

print(f"Word improvements: {details['improvements']['word'].mean():.2f}%")
print(f"Phrase improvements: {details['improvements']['phrase'].mean():.2f}%")
print(f"Sentence improvements: {details['improvements']['sentence'].mean():.2f}%")
```

## Comparison: Ray/OpenRLHF vs TRL/Unsloth

### When to use Ray + OpenRLHF
- ✅ Large-scale distributed training (multiple nodes)
- ✅ Very large models (70B+)
- ✅ Need vLLM for fast inference
- ✅ Have GPU cluster available

### When to use TRL + Unsloth
- ✅ Single GPU or small multi-GPU setup
- ✅ Want simple, maintainable code
- ✅ Need fast iteration and prototyping
- ✅ Working with models up to 13B (or 70B with quantization)
- ✅ Limited computational resources

### Feature Comparison

| Feature | Ray/OpenRLHF | TRL/Unsloth |
|---------|--------------|-------------|
| **Implementation** | `ray_utils_multilevel.py` | `multi_level_ppl_trl.py` |
| **Complexity** | High | Low |
| **Setup Time** | Hours | Minutes |
| **Code Lines** | ~700 | ~400 |
| **Dependencies** | Ray, DeepSpeed, OpenRLHF | TRL, Unsloth (optional) |
| **Distributed** | Yes (Ray) | Yes (Accelerate) |
| **Speed (single GPU)** | 1x | 2-5x (with Unsloth) |
| **Memory (8B model)** | ~32GB | ~8GB (4-bit) |

## Troubleshooting

### Issue: Out of Memory

```bash
# Solution 1: Use 4-bit quantization
--load_in_4bit

# Solution 2: Reduce batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16

# Solution 3: Enable gradient checkpointing
# (automatically enabled in the script)
```

### Issue: Slow Training

```bash
# Solution 1: Install Unsloth
pip install unsloth

# Solution 2: Use bfloat16
# (automatically enabled in the script)

# Solution 3: Increase batch size (if memory allows)
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2
```

### Issue: All Rewards are Zero

```python
# Cause: No baselines or baselines are too similar to current model

# Solution 1: Check baselines
with open('baselines.pkl', 'rb') as f:
    baselines = pickle.load(f)
    print(f"Num baselines: {len(baselines)}")
    sample = next(iter(baselines.values()))
    print(f"Sample baseline: {sample}")

# Solution 2: Use a different baseline model
reward_fn = MultiLevelPPLRewardFunction(
    baseline_model=different_model,  # Use a different model
    ...
)

# Solution 3: Lower reward thresholds
reward_thresholds = [
    (0.0, 0.1),  # Reward any improvement
    (0.5, 0.5),
    (1.0, 1.0),
]
```

### Issue: NaN Losses

```python
# Solution: Normalize rewards
reward_fn = MultiLevelPPLRewardFunction(
    ...,
    normalize=True,
)

# And use gradient clipping (default in GRPOConfig)
config = GRPOConfig(
    max_grad_norm=1.0,
)
```

## Advanced Usage

### Custom Reward Function

Extend the reward function for your specific use case:

```python
class CustomMultiLevelReward(MultiLevelPPLRewardFunction):
    def compute_rewards(self, texts, **kwargs):
        # Get base multi-level rewards
        base_rewards, details = super().compute_rewards(
            texts, return_details=True, **kwargs
        )

        # Add custom logic
        custom_rewards = []
        for text, base_reward in zip(texts, base_rewards):
            # Example: Bonus for longer responses
            length_bonus = min(len(text) / 1000, 0.2)

            # Example: Penalty for repetition
            words = text.split()
            unique_ratio = len(set(words)) / len(words) if words else 1.0
            repetition_penalty = (1 - unique_ratio) * 0.3

            final_reward = base_reward + length_bonus - repetition_penalty
            custom_rewards.append(final_reward)

        return torch.tensor(custom_rewards, device=self.device)
```

### Multi-Stage Training

Train with different reward configurations:

```python
# Stage 1: Focus on word-level accuracy
reward_fn_stage1 = MultiLevelPPLRewardFunction(
    ...,
    weights=(0.6, 0.3, 0.1),  # Emphasize words
)
trainer_stage1 = GRPOTrainer(..., reward_fn=reward_fn_stage1)
trainer_stage1.train()

# Stage 2: Focus on sentence coherence
reward_fn_stage2 = MultiLevelPPLRewardFunction(
    ...,
    weights=(0.1, 0.3, 0.6),  # Emphasize sentences
)
trainer_stage2 = GRPOTrainer(..., reward_fn=reward_fn_stage2)
trainer_stage2.train()
```

## Next Steps

1. ✅ Test the reward function: `python test_multilevel_ppl_trl.py`
2. ✅ Compute baselines for your dataset
3. ✅ Run a small training experiment (1 epoch, small dataset)
4. ✅ Monitor metrics and tune hyperparameters
5. ✅ Scale up to full training

## Resources

- **TRL Documentation**: https://huggingface.co/docs/trl
- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **GRPO Paper**: https://arxiv.org/abs/2402.03300
- **Multi-Level PPL Guide**: `MULTILEVEL_PPL_GUIDE.md`

## Citation

If you use this implementation in your research:

```bibtex
@software{multilevel_ppl_trl_2024,
  title={Multi-Level Perplexity Rewards for TRL and Unsloth},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ReasoningNCP}
}
```
