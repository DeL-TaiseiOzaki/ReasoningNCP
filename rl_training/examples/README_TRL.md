# TRL/Unsloth Examples

Examples for using multi-level perplexity rewards with TRL and Unsloth.

## Quick Start

### 1. Test the Implementation

```bash
# Test basic functionality
python test_multilevel_ppl_trl.py --model_name gpt2
```

Expected output:
```
Multi-Level PPL Reward Function Test (TRL)
...
✓ All tests passed!
```

### 2. Compute Baselines

```bash
# Compute baselines for your dataset
python compute_baselines_trl.py \
    --model_name gpt2 \
    --dataset_name wikitext \
    --dataset_split test \
    --output_path baselines.pkl \
    --num_samples 1000
```

### 3. Train with GRPO

```bash
# Train with multi-level PPL rewards
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --dataset_name your/dataset \
    --baselines_path baselines.pkl \
    --output_dir ./output
```

## File Descriptions

### test_multilevel_ppl_trl.py
- **Purpose**: Test TRL-compatible reward function
- **Usage**: `python test_multilevel_ppl_trl.py --model_name gpt2`
- **Output**: Verifies reward function works correctly

### compute_baselines_trl.py
- **Purpose**: Compute multi-level baselines for dataset
- **Usage**: `python compute_baselines_trl.py --model_name MODEL --dataset_name DATASET --output_path baselines.pkl`
- **Output**: Pickle file with per-sample multi-level PPLs

### train_grpo_multilevel_unsloth.py
- **Purpose**: Full GRPO training script with multi-level PPL rewards
- **Usage**: `python train_grpo_multilevel_unsloth.py [OPTIONS]`
- **Output**: Trained model with LoRA adapters

## Common Commands

### Test Different Models

```bash
# Test with GPT-2
python test_multilevel_ppl_trl.py --model_name gpt2

# Test with Llama 2
python test_multilevel_ppl_trl.py --model_name meta-llama/Llama-2-7b-hf

# Test with Mistral
python test_multilevel_ppl_trl.py --model_name mistralai/Mistral-7B-v0.1
```

### Compute Baselines for Different Datasets

```bash
# WikiText
python compute_baselines_trl.py \
    --model_name gpt2 \
    --dataset_name wikitext \
    --dataset_split test \
    --output_path baselines_wiki.pkl

# Custom dataset (local)
python compute_baselines_trl.py \
    --model_name gpt2 \
    --dataset_name ./my_dataset \
    --text_key content \
    --output_path baselines_custom.pkl

# Large dataset (sample 5000)
python compute_baselines_trl.py \
    --model_name gpt2 \
    --dataset_name openwebtext \
    --num_samples 5000 \
    --batch_size 8 \
    --output_path baselines_owt.pkl
```

### Training Configurations

#### Fast Prototyping (Single GPU)
```bash
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --dataset_name your/dataset \
    --baselines_path baselines.pkl \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_samples 1000 \
    --output_dir ./output_test
```

#### Production Training
```bash
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --dataset_name your/dataset \
    --baselines_path baselines.pkl \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --wandb_project my-project \
    --output_dir ./output_prod
```

#### Memory-Constrained
```bash
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/mistral-7b-bnb-4bit \
    --dataset_name your/dataset \
    --baselines_path baselines.pkl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --load_in_4bit \
    --output_dir ./output_low_mem
```

#### Custom PPL Weights
```bash
# Emphasize sentence-level coherence (for story generation)
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --dataset_name your/dataset \
    --baselines_path baselines.pkl \
    --ppl_weights 0.1 0.3 0.6 \
    --output_dir ./output_stories

# Emphasize word-level accuracy (for technical writing)
python train_grpo_multilevel_unsloth.py \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --dataset_name your/dataset \
    --baselines_path baselines.pkl \
    --ppl_weights 0.4 0.4 0.2 \
    --output_dir ./output_technical
```

## Installation

### Minimum Requirements

```bash
pip install torch transformers accelerate trl
```

### With Unsloth (Recommended, 2-5x Faster)

```bash
# For CUDA 12.1
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8
pip install "unsloth[cu118-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

### With WandB Logging

```bash
pip install wandb
wandb login
```

## Troubleshooting

### Out of Memory

```bash
# Use smaller batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16

# Use 4-bit quantization (if not already)
--load_in_4bit
```

### Import Errors

```bash
# Ensure you're in the examples directory
cd rl_training/examples

# Or add parent to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
```

### Slow Training

```bash
# Install Unsloth for 2-5x speedup
pip install unsloth

# Use larger batch size (if memory allows)
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2
```

### All Rewards are Zero

```bash
# Check baselines file
python -c "import pickle; f=open('baselines.pkl','rb'); d=pickle.load(f); print(len(d), list(d.values())[0])"

# Lower reward thresholds in the code
# Or compute baselines with the same model being trained
```

## Expected Results

### Test Script Output

```
Multi-Level PPL Reward Function Test (TRL)
============================================================
Model: gpt2
Device: cuda

Testing Multi-Level PPL Calculator
...
Results:
  Word-level PPL:     45.23
  Phrase-level PPL:   47.89
  Sentence-level PPL: 51.21

Testing Reward Function
...
Rewards (with mock baselines):
1. Prompt: The quick brown fox
   Improvements: Word=12.34%, Phrase=10.56%, Sentence=14.23%
   Average improvement: 12.51%
   Reward: 1.0000

✓ All tests passed!
```

### Baseline Computation Output

```
Multi-Level Baseline Computation (TRL)
============================================================
Model:        gpt2
Dataset:      wikitext
...

Processing 1000 samples in batches of 4...
100%|████████████████████| 250/250 [05:23<00:00,  0.77it/s]

Statistics:
============================================================
Samples processed: 1000

Word-level PPL:
  Mean:   42.15
  Min:    12.35
  Max:    125.79
  Median: 38.90

Phrase-level PPL:
  Mean:   44.23
  Min:    13.89
  Max:    132.46
  Median: 40.78

Sentence-level PPL:
  Mean:   47.89
  Min:    15.23
  Max:    145.68
  Median: 44.12

Done!
```

### Training Output

```
GRPO Training with Multi-Level PPL Rewards
============================================================
Model: unsloth/llama-3-8b-bnb-4bit
Dataset: your/dataset
...

Loading model with Unsloth...
Model loaded successfully!

Initializing multi-level PPL reward function...
Reward function initialized!

Starting training...
============================================================

  0%|                                        | 0/250 [00:00<?, ?it/s]
{'loss': 0.234, 'learning_rate': 5e-5, 'epoch': 0.1}
 10%|███▋                              | 25/250 [02:15<20:18,  5.42s/it]
...

Training complete!
Saving model to ./output/final...
Done!
```

## Performance Benchmarks

| Setup | Training Speed | Memory Usage | Notes |
|-------|---------------|--------------|-------|
| GPT-2 (standard) | 1x | 4GB | Baseline |
| GPT-2 + Unsloth | 2.3x | 3GB | Faster |
| Llama 3-8B (fp16) | 1x | 28GB | Large |
| Llama 3-8B (4-bit) | 0.8x | 8GB | Memory-efficient |
| Llama 3-8B (4-bit + Unsloth) | 2.1x | 7GB | **Recommended** |

## Next Steps

1. ✅ Run `test_multilevel_ppl_trl.py` to verify setup
2. ✅ Compute baselines for your dataset
3. ✅ Run a small training experiment (100 samples, 1 epoch)
4. ✅ Monitor metrics and tune hyperparameters
5. ✅ Scale up to full training

## Additional Resources

- **Complete Guide**: See `../TRL_UNSLOTH_GUIDE.md` for detailed documentation
- **TRL Docs**: https://huggingface.co/docs/trl
- **Unsloth**: https://github.com/unslothai/unsloth
- **Multi-Level PPL Theory**: See `../MULTILEVEL_PPL_GUIDE.md`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. See the full guide: `../TRL_UNSLOTH_GUIDE.md`
3. Open an issue on GitHub
