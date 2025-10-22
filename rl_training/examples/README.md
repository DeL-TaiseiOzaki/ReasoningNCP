# Multi-Level Perplexity Examples

This directory contains example scripts for using the multi-level perplexity reward function.

## Quick Start

### 1. Test Multi-Level PPL Calculation

Test the multi-level perplexity calculator with sample texts:

```bash
# Basic test with default texts
python test_multilevel_ppl.py --model_name gpt2

# Test with custom text
python test_multilevel_ppl.py \
    --model_name gpt2 \
    --text "Your test text here." \
    --baseline_text "Your baseline text here."

# Use a different model
python test_multilevel_ppl.py --model_name meta-llama/Llama-2-7b-hf
```

**Expected output:**
```
Multi-Level Perplexity Test
============================================================

Model: gpt2
Device: cuda

Loading model and tokenizer...
Model loaded successfully!

============================================================
TEST TEXT EVALUATION
============================================================

Results:
------------------------------------------------------------
Word-level PPL:          45.2341
Phrase-level PPL:        47.8932
Sentence-level PPL:      51.2104

Phrase-level breakdown:
  phrase_2gram: PPL = 46.1234 (15 chunks)
  phrase_3gram: PPL = 48.5629 (10 chunks)
  phrase_4gram: PPL = 49.8932 (7 chunks)

Weighted aggregate:      48.9876

============================================================
BASELINE TEXT EVALUATION
============================================================

Results:
------------------------------------------------------------
Word-level PPL:          48.5123
Phrase-level PPL:        50.2341
Sentence-level PPL:      54.3210

============================================================
REWARD CALCULATION
============================================================

Comparison (Test vs Baseline):
------------------------------------------------------------
Word improvement:           6.76%
Phrase improvement:         4.67%
Sentence improvement:       5.72%

Weighted average:           5.58%

Reward:                    1.0000

============================================================
Interpretation:
------------------------------------------------------------
✓ Test text has 5.58% better perplexity than baseline
✓ Reward score: 1.0000
============================================================
```

### 2. Compute Baseline Perplexities

Compute multi-level baselines for your dataset:

```bash
# Example with a HuggingFace dataset
python compute_baseline_multilevel.py \
    --model_name gpt2 \
    --dataset_path wikitext \
    --dataset_split test \
    --text_key text \
    --output_path baselines_wikitext.pkl \
    --num_samples 1000

# Example with a local dataset
python compute_baseline_multilevel.py \
    --model_name your-model-path \
    --dataset_path /path/to/your/dataset \
    --dataset_split train \
    --text_key content \
    --id_key sample_id \
    --output_path baselines_custom.pkl \
    --num_samples 5000 \
    --phrase_sizes 2 3 4 5
```

**Expected output:**
```
================================================================================
Multi-Level Baseline Perplexity Computation
================================================================================

Model:           gpt2
Dataset:         wikitext
Split:           test
Output:          baselines_wikitext.pkl
Max samples:     1000
Device:          cuda
Phrase sizes:    [2, 3, 4]
--------------------------------------------------------------------------------

Loading model and tokenizer...
Model loaded successfully!

Loading dataset from wikitext...
Dataset loaded: 1000 samples

================================================================================
Computing baselines...
================================================================================

Processing 1000 samples...
100%|███████████████████████████████████████| 1000/1000 [05:23<00:00,  3.09it/s]

  Avg PPL so far - Word: 42.15, Phrase: 44.23, Sentence: 47.89

================================================================================
Statistics:
================================================================================

Samples processed: 1000

Word-level PPL:
  Mean:   42.1543
  Min:    12.3456
  Max:    125.7890
  Median: 38.9012

Phrase-level PPL:
  Mean:   44.2341
  Min:    13.8901
  Max:    132.4567
  Median: 40.7823

Sentence-level PPL:
  Mean:   47.8912
  Min:    15.2345
  Max:    145.6789
  Median: 44.1234

Saving baselines to baselines_wikitext.pkl...
Done!
================================================================================
```

### 3. Using Baselines in Your Training Script

Load and use the computed baselines:

```python
import pickle

# Load baselines
with open('baselines_wikitext.pkl', 'rb') as f:
    baselines = pickle.load(f)

# Get baseline for a specific sample
sample_id = 'sample_123'
baseline_ppls = {
    'word': baselines[sample_id]['word'],
    'phrase': baselines[sample_id]['phrase'],
    'sentence': baselines[sample_id]['sentence'],
}

print(f"Baseline PPLs for {sample_id}:")
print(f"  Word: {baseline_ppls['word']:.2f}")
print(f"  Phrase: {baseline_ppls['phrase']:.2f}")
print(f"  Sentence: {baseline_ppls['sentence']:.2f}")
```

## File Descriptions

### test_multilevel_ppl.py
- **Purpose**: Test the multi-level PPL calculator with sample texts
- **Use case**: Quick verification, debugging, comparing texts
- **Output**: Detailed PPL breakdown and reward calculation

### compute_baseline_multilevel.py
- **Purpose**: Compute baselines for an entire dataset
- **Use case**: Pre-compute baselines before training
- **Output**: Pickle file with per-sample multi-level PPLs

## Common Parameters

### Model Selection
```bash
--model_name gpt2                    # Small, fast, for testing
--model_name meta-llama/Llama-2-7b-hf  # Larger, better quality
--model_name /path/to/local/model    # Your custom model
```

### Device
```bash
--device cuda      # Use GPU (default if available)
--device cpu       # Use CPU only
--device cuda:0    # Specific GPU
```

### Phrase Configuration
```bash
--phrase_sizes 2 3 4       # Default: 2-gram, 3-gram, 4-gram
--phrase_sizes 2 3         # Shorter phrases
--phrase_sizes 3 4 5 6     # Longer phrases
```

## Troubleshooting

### Out of Memory (OOM)
```bash
# Use smaller model
python test_multilevel_ppl.py --model_name gpt2

# Process fewer samples
python compute_baseline_multilevel.py --num_samples 100

# Use CPU
python compute_baseline_multilevel.py --device cpu
```

### Import Errors
```bash
# Make sure you're in the examples directory
cd rl_training/examples

# Or add parent directory to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
```

### Slow Processing
```bash
# Use GPU if available
python compute_baseline_multilevel.py --device cuda

# Process fewer samples for testing
python compute_baseline_multilevel.py --num_samples 100

# Use smaller n-gram sizes
python compute_baseline_multilevel.py --phrase_sizes 2 3
```

## Advanced Usage

### Custom Reward Thresholds

Edit `test_multilevel_ppl.py` to customize reward thresholds:

```python
reward_thresholds = [
    (0.1, 0.2),    # >= 0.1% improvement -> 0.2 reward
    (1.0, 0.5),    # >= 1.0% improvement -> 0.5 reward
    (5.0, 1.0),    # >= 5.0% improvement -> 1.0 reward
]
```

### Custom Weights

Adjust importance of each level:

```python
# Emphasize sentence-level coherence
weights = (0.1, 0.3, 0.6)

# Emphasize word-level accuracy
weights = (0.5, 0.3, 0.2)

# Equal weighting
weights = (0.33, 0.33, 0.34)
```

### Custom Sentence Delimiters

For different languages or formats:

```python
# Default (English)
sentence_delimiters = r'[.!?。！？]'

# Add newlines
sentence_delimiters = r'[.!?\n。！？]'

# Chinese/Japanese focus
sentence_delimiters = r'[。！？]'
```

## Next Steps

1. **Test with your data**: Run `test_multilevel_ppl.py` with your actual texts
2. **Compute baselines**: Run `compute_baseline_multilevel.py` on your dataset
3. **Integrate into training**: Use `ray_utils_multilevel.py` in your training script
4. **Monitor results**: Track multi-level improvements during training

For more details, see the main documentation: [MULTILEVEL_PPL_GUIDE.md](../MULTILEVEL_PPL_GUIDE.md)
