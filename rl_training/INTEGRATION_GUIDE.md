# Integration Guide: Multi-Level PPL into Existing Training

This guide shows how to integrate the multi-level perplexity reward function into your existing training pipeline.

## Quick Integration (3 Steps)

### Step 1: Import the Multi-Level Components

In your `train_ncp.py` (or equivalent training script):

```python
# OLD:
from ray_utils import CustomStoryBasedActorModelRayActor

# NEW: Add this import
from ray_utils_multilevel import MultiLevelPPLActorModelRayActor
```

### Step 2: Use the Multi-Level Actor

Replace the actor class:

```python
# OLD:
actor_model = PPORayActorGroup(
    args.actor_num_nodes,
    args.actor_num_gpus_per_node,
    CustomStoryBasedActorModelRayActor,  # ← Old version
    pg=pg,
    num_gpus_per_actor=0.2 if pg else 1,
)

# NEW:
actor_model = PPORayActorGroup(
    args.actor_num_nodes,
    args.actor_num_gpus_per_node,
    MultiLevelPPLActorModelRayActor,  # ← Multi-level version
    pg=pg,
    num_gpus_per_actor=0.2 if pg else 1,
)
```

### Step 3: Pre-compute Multi-Level Baselines

Before training, compute baselines using the script:

```bash
python examples/compute_baseline_multilevel.py \
    --model_name your-baseline-model \
    --dataset_path your-dataset \
    --output_path baseline_multilevel_ppls.pkl \
    --num_samples 10000
```

Then update your data loading to include multi-level baselines (see below).

## Detailed Integration Steps

### 1. Update Baseline Data Loading

The existing code loads single-level baselines:

```python
# ray_utils.py (OLD)
with open("/mnt/disk/nrl_ncp/prompt_to_datapoint_with_baseline_ppl_qwen3B.pkl", "rb") as f:
    prompt_to_datapoint_with_baseline_ppl = pickle.load(f)
```

You need to either:

**Option A: Extend existing baselines** (Recommended)

Add multi-level PPLs to your existing baseline pickle file:

```python
import pickle
from multi_level_ppl import MultiLevelPerplexityCalculator

# Load existing baselines
with open("prompt_to_datapoint_with_baseline_ppl_qwen3B.pkl", "rb") as f:
    data = pickle.load(f)

# Add multi-level PPLs
calculator = MultiLevelPerplexityCalculator(tokenizer)

for key, datapoint in data.items():
    # Assume datapoint has 'baseline_ppl' (single-level)
    single_ppl = datapoint['baseline_ppl'].item()

    # Approximate multi-level from single-level
    # (In production, compute actual multi-level PPLs)
    datapoint['baseline_ppl_word'] = single_ppl
    datapoint['baseline_ppl_phrase'] = single_ppl * 1.1
    datapoint['baseline_ppl_sentence'] = single_ppl * 1.2

# Save updated baselines
with open("prompt_to_datapoint_with_baseline_ppl_multilevel.pkl", "wb") as f:
    pickle.dump(data, f)
```

**Option B: Create new baselines**

Compute multi-level baselines from scratch using the example script.

### 2. Modify Data Preparation

Update `get_next_chapter_messages_from_sequence` if needed:

```python
# ray_utils_multilevel.py already handles this
# The function now returns multi-level baselines

def get_next_chapter_messages_from_sequence(sequence, tokenizer):
    # ... existing code ...

    datapoint = prompt_to_datapoint_with_baseline_ppl[next_chapter_synopsis]

    # OLD: Single baseline
    # baseline_ppl = datapoint["baseline_ppl"].detach().to("cpu").item()

    # NEW: Multi-level baselines
    baseline_ppl_word = datapoint.get("baseline_ppl_word", datapoint["baseline_ppl"]).detach().to("cpu").item()
    baseline_ppl_phrase = datapoint.get("baseline_ppl_phrase", baseline_ppl_word * 1.1).detach().to("cpu").item()
    baseline_ppl_sentence = datapoint.get("baseline_ppl_sentence", baseline_ppl_word * 1.2).detach().to("cpu").item()

    # ... rest of code ...

    return (next_chapter_tokens, labels,
            (baseline_ppl_word, baseline_ppl_phrase, baseline_ppl_sentence),
            model_response, original_model_response, next_chapter_text)
```

### 3. Configure Reward Weights and Thresholds

Customize in `ray_utils_multilevel.py`:

```python
class MultiLevelPPLRemoteExperienceMaker(RemoteExperienceMaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Customize weights: (word, phrase, sentence)
        self.ppl_weights = (0.2, 0.3, 0.5)  # Default

        # For your task, you might want:
        # Story generation -> emphasize sentence coherence
        # self.ppl_weights = (0.1, 0.3, 0.6)

        # Technical writing -> emphasize word precision
        # self.ppl_weights = (0.4, 0.4, 0.2)

        # Customize reward thresholds
        self.reward_thresholds = [
            (0.05, 0.2),   # Small improvement
            (0.5, 0.5),    # Medium improvement
            (1.0, 0.7),    # Good improvement
            (2.0, 0.9),    # Great improvement
            (3.0, 1.0),    # Excellent improvement
        ]
```

### 4. Update Logging and Monitoring

The multi-level version provides additional metrics. Update your logging:

```python
# In your training loop or callback
def log_metrics(info):
    # OLD: Single reward
    # wandb.log({'reward': info['reward'].mean()})

    # NEW: Multi-level metrics
    wandb.log({
        'reward/total': info['reward'].mean(),
        'reward/word_improvement': info['word_improvements'].mean(),
        'reward/phrase_improvement': info['phrase_improvements'].mean(),
        'reward/sentence_improvement': info['sentence_improvements'].mean(),
        'reward/avg_improvement': info['avg_improvements'].mean(),
        'ppl/word': info['word_ppls'].mean(),
        'ppl/phrase': info['phrase_ppls'].mean(),
        'ppl/sentence': info['sentence_ppls'].mean(),
    })
```

## Side-by-Side Comparison

### Original Implementation (Single-Level PPL)

```python
# ray_utils.py
class StoryBasedRemoteExperienceMaker(RemoteExperienceMaker):
    @torch.no_grad()
    def get_r_refs(self, sequences_cpu, ...):
        rewards = []
        for sequence in sequences_cpu:
            # ... prepare inputs ...

            loss = self.initial_model.forward.remote(...)
            loss = ray.get([loss])[0]
            perplexity = torch.exp(loss).item()

            # Single-level comparison
            percent_improvement = (baseline_ppl - perplexity) / baseline_ppl * 100

            # Simple threshold-based reward
            reward = 0.5 if percent_improvement >= 0.05 else 0
            reward = 0.9 if percent_improvement >= 1 else reward
            reward = 1.0 if percent_improvement >= 2 else reward

            rewards.append(torch.tensor([reward]))

        return rewards, pct_improvements
```

### Multi-Level Implementation

```python
# ray_utils_multilevel.py
class MultiLevelPPLRemoteExperienceMaker(RemoteExperienceMaker):
    @torch.no_grad()
    def get_r_refs(self, sequences_cpu, ...):
        rewards = []
        multi_level_info = {...}

        for sequence in sequences_cpu:
            # ... prepare inputs ...

            # Compute word-level PPL
            word_loss = self.initial_model.forward.remote(...)
            word_ppl = torch.exp(ray.get([word_loss])[0]).item()

            # Compute phrase-level PPL
            phrase_ppl = ...  # (from same forward pass)

            # Compute sentence-level PPL
            sentence_ppl = ...  # (from same forward pass)

            # Multi-level comparison
            baseline_ppls = {'word': ..., 'phrase': ..., 'sentence': ...}
            current_ppls = {'word': word_ppl, 'phrase': phrase_ppl, 'sentence': sentence_ppl}

            # Weighted reward based on all levels
            reward, info = compute_multi_level_reward(
                baseline_ppls, current_ppls,
                weights=(0.2, 0.3, 0.5),
                reward_thresholds=[...]
            )

            rewards.append(torch.tensor([reward]))
            # Store detailed info for logging
            multi_level_info['word_improvements'].append(info['improvements']['word'])
            # ... etc ...

        return rewards, multi_level_info
```

## Testing Your Integration

### 1. Dry Run Test

Test without full training:

```python
# test_integration.py
import ray
from ray_utils_multilevel import MultiLevelPPLRemoteExperienceMaker
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize Ray
ray.init()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("your-model")
model = AutoModelForCausalLM.from_pretrained("your-model")

# Create experience maker
experience_maker = MultiLevelPPLRemoteExperienceMaker(
    actor=model,
    critic=None,
    reward_model=None,
    initial_model=model,
    tokenizer=tokenizer,
    # ... other args ...
)

# Test with dummy data
dummy_sequences = torch.randint(0, 1000, (4, 512))
dummy_attention_mask = torch.ones_like(dummy_sequences)

rewards, info = experience_maker.get_r_refs(
    dummy_sequences, dummy_attention_mask, None, 512
)

print(f"Rewards: {rewards}")
print(f"Word improvements: {info['word_improvements']}")
print(f"Phrase improvements: {info['phrase_improvements']}")
print(f"Sentence improvements: {info['sentence_improvements']}")
```

### 2. Single Batch Test

Run training for 1 batch:

```bash
# Modify train_ncp.py to add --max_steps 1
python train_ncp.py \
    --pretrain your-model \
    --prompt_data your-data \
    --num_episodes 1 \
    --rollout_batch_size 8 \
    --max_steps 1  # Stop after 1 batch
```

### 3. Monitor Outputs

Check that multi-level metrics are logged:

```bash
# In your logs, you should see:
# reward/word_improvement: 1.23
# reward/phrase_improvement: 0.98
# reward/sentence_improvement: 1.45
# ppl/word: 42.15
# ppl/phrase: 44.23
# ppl/sentence: 47.89
```

## Rollback Plan

If you encounter issues, you can easily rollback:

```python
# Simply change back to:
from ray_utils import CustomStoryBasedActorModelRayActor

actor_model = PPORayActorGroup(
    args.actor_num_nodes,
    args.actor_num_gpus_per_node,
    CustomStoryBasedActorModelRayActor,  # Back to original
    pg=pg,
    num_gpus_per_actor=0.2 if pg else 1,
)
```

## Performance Considerations

### Computational Overhead

- **Single-level**: 1× cost
- **Multi-level**: ~1.2× cost (10-20% overhead)

The overhead is minimal because phrase and sentence-level PPLs are computed from the same forward pass.

### Memory Usage

No significant memory overhead - same model, same batch size.

### Training Time

Expect ~10-20% increase in training time per episode.

## Common Issues and Solutions

### Issue 1: Missing baseline keys

```
KeyError: 'baseline_ppl_word'
```

**Solution**: Add fallback to single-level baseline:

```python
baseline_word = datapoint.get('baseline_ppl_word',
                               datapoint['baseline_ppl']).item()
```

### Issue 2: NaN rewards

```
RuntimeError: reward is NaN
```

**Solution**: Check baseline values aren't zero or infinity:

```python
if baseline_ppl <= 0 or baseline_ppl == float('inf'):
    baseline_ppl = 10.0  # Fallback value
```

### Issue 3: All rewards are zero

**Solution**: Lower reward thresholds or check baseline quality:

```python
# More lenient thresholds
reward_thresholds = [
    (0.0, 0.1),    # Any improvement
    (0.5, 0.5),
    (1.0, 1.0),
]
```

## Best Practices

1. **Always compute baselines first** - Don't start training without proper baselines
2. **Monitor all levels** - Track word, phrase, and sentence improvements separately
3. **Tune weights for your task** - Different tasks may need different level importance
4. **Start with conservative thresholds** - You can make them stricter later
5. **Compare with single-level** - Run parallel experiments to verify improvement

## Next Steps

1. ✅ Integrate multi-level PPL into your training script
2. ✅ Compute multi-level baselines for your dataset
3. ✅ Run a test training with 1-2 episodes
4. ✅ Monitor multi-level metrics
5. ✅ Tune weights and thresholds based on results
6. ✅ Scale up to full training

For questions or issues, refer to:
- [MULTILEVEL_PPL_GUIDE.md](MULTILEVEL_PPL_GUIDE.md) - Detailed documentation
- [examples/README.md](examples/README.md) - Example scripts
- Original paper references in the guide

Good luck with your multi-level PPL training!
