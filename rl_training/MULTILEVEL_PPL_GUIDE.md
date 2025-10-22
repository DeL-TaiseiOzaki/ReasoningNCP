# Multi-Level Perplexity Reward Function Guide

## Overview

This guide explains how to use the multi-level perplexity (PPL) reward function for reinforcement learning from human feedback (RLHF) training.

## What is Multi-Level Perplexity?

Instead of evaluating model outputs at a single granularity, we measure perplexity at three levels:

### 1. **Word-Level Perplexity**
- Evaluates individual token predictions
- Measures: How well the model predicts each next token
- **Formula**: `PPL_word = exp(mean(-log P(token_i | context)))`
- **Interpretation**: Lower is better; indicates basic language modeling capability

### 2. **Phrase-Level Perplexity**
- Evaluates multi-token phrases (n-grams)
- Default: 2-grams, 3-grams, and 4-grams
- Measures: Coherence of short multi-word expressions
- **Interpretation**: Captures local coherence beyond single tokens

### 3. **Sentence-Level Perplexity**
- Evaluates complete sentences or utterances
- Measures: Long-range consistency and fluency
- **Interpretation**: Captures global coherence across full thoughts

## Why Multi-Level?

Single-level perplexity can miss important aspects:
- Word-level alone may reward models that predict individual words well but produce incoherent phrases
- Sentence-level alone may not catch local phrase-level errors
- **Multi-level captures both local and global quality**

## Architecture

```
┌─────────────────────────────────────────┐
│  Multi-Level PPL Reward Calculation     │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────┐  Weight: 0.2         │
│  │ Word-Level    │  (Individual tokens) │
│  │ PPL = 8.5     │                      │
│  └───────────────┘                      │
│           │                             │
│           ├──────────────┐              │
│           │              │              │
│  ┌───────────────┐      │              │
│  │ Phrase-Level  │  Weight: 0.3         │
│  │ PPL = 9.2     │  (2-4 word phrases) │
│  └───────────────┘      │              │
│           │              │              │
│           ├──────────────┤              │
│           │              │              │
│  ┌───────────────┐      │              │
│  │ Sentence-Level│  Weight: 0.5         │
│  │ PPL = 10.1    │  (Full utterances)  │
│  └───────────────┘                      │
│           │                             │
│           ▼                             │
│  ┌─────────────────────┐               │
│  │ Weighted Aggregation│               │
│  │ Avg PPL = 9.4       │               │
│  └─────────────────────┘               │
│           │                             │
│           ▼                             │
│  ┌─────────────────────┐               │
│  │ Compare to Baseline │               │
│  │ Improvement = 1.5%  │               │
│  └─────────────────────┘               │
│           │                             │
│           ▼                             │
│  ┌─────────────────────┐               │
│  │  Reward = 0.7       │               │
│  └─────────────────────┘               │
└─────────────────────────────────────────┘
```

## File Structure

```
rl_training/
├── multi_level_ppl.py              # Core PPL calculation module
├── ray_utils_multilevel.py         # Multi-level PPL experience maker
├── train_ncp_multilevel.py         # Training script (to be created)
├── MULTILEVEL_PPL_GUIDE.md         # This file
└── examples/
    ├── compute_baseline_multilevel.py   # Compute baseline PPLs
    └── test_multilevel_ppl.py           # Test PPL calculations
```

## Usage

### 1. Basic Multi-Level PPL Calculation

```python
from multi_level_ppl import MultiLevelPerplexityCalculator

# Initialize calculator
calculator = MultiLevelPerplexityCalculator(
    tokenizer=tokenizer,
    phrase_sizes=[2, 3, 4],  # n-gram sizes for phrase-level
    sentence_delimiters=r'[.!?。！？]'  # Sentence splitting pattern
)

# Compute all levels at once
results = calculator.compute_all_levels(
    model=model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    text=decoded_text  # Optional, for better sentence splitting
)

print(f"Word-level PPL: {results['word']['perplexity']:.2f}")
print(f"Phrase-level PPL: {results['phrase']['perplexity']:.2f}")
print(f"Sentence-level PPL: {results['sentence']['perplexity']:.2f}")
```

### 2. Computing Rewards from Multi-Level PPL

```python
from multi_level_ppl import compute_multi_level_reward

# Define baselines (pre-computed on validation set)
baseline_ppls = {
    'word': 10.5,
    'phrase': 11.2,
    'sentence': 12.8
}

# Current model's perplexities
current_ppls = {
    'word': 9.8,
    'phrase': 10.5,
    'sentence': 11.9
}

# Compute reward
reward, info = compute_multi_level_reward(
    baseline_ppls=baseline_ppls,
    current_ppls=current_ppls,
    weights=(0.2, 0.3, 0.5),  # word, phrase, sentence weights
    reward_thresholds=[
        (0.05, 0.2),   # >= 0.05% improvement -> 0.2 reward
        (0.5, 0.5),    # >= 0.5% improvement -> 0.5 reward
        (1.0, 0.7),    # >= 1% improvement -> 0.7 reward
        (2.0, 0.9),    # >= 2% improvement -> 0.9 reward
        (3.0, 1.0),    # >= 3% improvement -> 1.0 reward
    ]
)

print(f"Reward: {reward:.3f}")
print(f"Word improvement: {info['improvements']['word']:.2f}%")
print(f"Phrase improvement: {info['improvements']['phrase']:.2f}%")
print(f"Sentence improvement: {info['improvements']['sentence']:.2f}%")
print(f"Average improvement: {info['avg_improvement']:.2f}%")
```

### 3. Training with Multi-Level PPL

Modify your training script to use the multi-level version:

```python
from ray_utils_multilevel import MultiLevelPPLActorModelRayActor

# In train_ncp.py, replace:
# from ray_utils import CustomStoryBasedActorModelRayActor

# With:
# from ray_utils_multilevel import MultiLevelPPLActorModelRayActor

actor_model = PPORayActorGroup(
    args.actor_num_nodes,
    args.actor_num_gpus_per_node,
    MultiLevelPPLActorModelRayActor,  # Use multi-level version
    pg=pg,
    num_gpus_per_actor=0.2 if pg else 1,
)
```

## Configuration

### Adjusting Level Weights

The default weights are `(0.2, 0.3, 0.5)` for word, phrase, and sentence levels respectively. You can adjust these based on your task:

```python
# For tasks requiring high fluency (e.g., story generation)
weights = (0.1, 0.3, 0.6)  # Emphasize sentence-level coherence

# For tasks requiring precise wording (e.g., technical writing)
weights = (0.4, 0.4, 0.2)  # Emphasize word and phrase accuracy

# Balanced approach
weights = (0.33, 0.33, 0.34)  # Equal importance
```

### Adjusting Reward Thresholds

Customize reward thresholds based on your baseline perplexity and desired sensitivity:

```python
# More granular rewards for small improvements
reward_thresholds = [
    (0.01, 0.1),
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
```

### Phrase Size Configuration

Adjust n-gram sizes for phrase-level evaluation:

```python
# Shorter phrases (better for chat/dialogue)
phrase_sizes = [2, 3]

# Longer phrases (better for formal text)
phrase_sizes = [3, 4, 5]

# Mixed range
phrase_sizes = [2, 3, 4, 5, 6]
```

## Computing Baselines

Before training, compute baseline perplexities on your validation set:

```python
# See examples/compute_baseline_multilevel.py for full script

from multi_level_ppl import MultiLevelPerplexityCalculator
import pickle

calculator = MultiLevelPerplexityCalculator(tokenizer)

baselines = {}
for datapoint in validation_dataset:
    input_ids, attention_mask, labels = prepare_inputs(datapoint)

    results = calculator.compute_all_levels(
        model=baseline_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    baselines[datapoint['id']] = {
        'word': results['word']['perplexity'],
        'phrase': results['phrase']['perplexity'],
        'sentence': results['sentence']['perplexity'],
    }

# Save baselines
with open('baseline_multilevel_ppls.pkl', 'wb') as f:
    pickle.dump(baselines, f)
```

## Monitoring Training

The multi-level PPL experience maker logs detailed metrics:

```python
# Available in training logs:
- word_improvements: Per-sample word-level improvements
- phrase_improvements: Per-sample phrase-level improvements
- sentence_improvements: Per-sample sentence-level improvements
- avg_improvements: Weighted average improvements
- word_ppls: Current word-level perplexities
- phrase_ppls: Current phrase-level perplexities
- sentence_ppls: Current sentence-level perplexities
```

Example WandB logging:

```python
wandb.log({
    'reward/word_improvement': info['word_improvements'].mean(),
    'reward/phrase_improvement': info['phrase_improvements'].mean(),
    'reward/sentence_improvement': info['sentence_improvements'].mean(),
    'reward/avg_improvement': info['avg_improvements'].mean(),
    'ppl/word': info['word_ppls'].mean(),
    'ppl/phrase': info['phrase_ppls'].mean(),
    'ppl/sentence': info['sentence_ppls'].mean(),
})
```

## Performance Considerations

### Computational Cost

Multi-level PPL requires multiple passes or detailed analysis:
- **Word-level**: Same as standard PPL (1 forward pass)
- **Phrase-level**: Computed from same forward pass (minimal overhead)
- **Sentence-level**: Computed from same forward pass (minimal overhead)

**Total overhead**: ~10-20% compared to single-level PPL

### Optimization Tips

1. **Batch computation**: Process multiple samples together
2. **Caching**: Cache logits if computing multiple PPL levels
3. **Approximation**: For very long sequences, subsample sentences
4. **Parallelization**: Use Ray for distributed PPL computation

## Troubleshooting

### Issue: Phrase-level PPL is NaN

**Cause**: Sequence too short for n-gram size

**Solution**:
```python
calculator = MultiLevelPerplexityCalculator(
    tokenizer=tokenizer,
    phrase_sizes=[2, 3],  # Use smaller n-grams
)
```

### Issue: Sentence-level PPL equals word-level

**Cause**: No sentence delimiters found in text

**Solution**: Provide decoded text or adjust delimiter pattern
```python
calculator = MultiLevelPerplexityCalculator(
    tokenizer=tokenizer,
    sentence_delimiters=r'[.!?\n。！？]'  # Add more delimiters
)
```

### Issue: All rewards are 0

**Cause**: Baselines may be too optimistic or current model is worse

**Solution**: Check baseline values and adjust thresholds
```python
# Lower the minimum threshold
reward_thresholds = [
    (0.0, 0.1),    # Any improvement gets some reward
    (0.5, 0.5),
    (1.0, 1.0),
]
```

## Comparison with Single-Level PPL

| Aspect | Single-Level PPL | Multi-Level PPL |
|--------|-----------------|-----------------|
| **Granularity** | One level (usually word) | Three levels |
| **Coherence** | Token-level only | Local + global |
| **Sensitivity** | May miss phrase errors | Catches phrase + sentence issues |
| **Computation** | 1× cost | ~1.2× cost |
| **Interpretability** | Simple | Rich, detailed |
| **Training signal** | Basic | Multi-faceted |

## References

- [Perplexity in NLP - Baeldung](https://www.baeldung.com/cs/nlp-perplexity)
- [Language Model Evaluation Metrics](https://arxiv.org/abs/1904.09675)
- [Multi-granularity Language Modeling](https://arxiv.org/abs/2106.12062)

## Citation

If you use this multi-level PPL approach in your research, please cite:

```bibtex
@software{multilevel_ppl_2024,
  title={Multi-Level Perplexity for RLHF},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ReasoningNCP}
}
```

## License

Same as the parent ReasoningNCP project.
