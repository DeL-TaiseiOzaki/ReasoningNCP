"""
Multi-Level Perplexity Reward Function for TRL and Unsloth

This module provides a TRL-compatible reward function for GRPO (Group Relative Policy Optimization)
training with multi-level perplexity evaluation.

Compatible with:
- TRL (Transformers Reinforcement Learning) GRPOTrainer
- Unsloth (fast training library)
- Standard HuggingFace models

Usage:
    from multi_level_ppl_trl import MultiLevelPPLRewardFunction

    reward_fn = MultiLevelPPLRewardFunction(
        baseline_model=model,
        tokenizer=tokenizer,
        baselines_path="baselines.pkl"
    )

    # Use with TRL GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_fn=reward_fn,
        ...
    )
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable
import pickle
import re
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class MultiLevelPPLCalculator:
    """
    Lightweight multi-level perplexity calculator optimized for TRL/Unsloth.

    This version is more efficient and doesn't require Ray or distributed computing.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        phrase_sizes: List[int] = [2, 3, 4],
        sentence_delimiters: str = r'[.!?。！？\n]',
        device: str = 'cuda'
    ):
        self.tokenizer = tokenizer
        self.phrase_sizes = phrase_sizes
        self.sentence_delimiter_pattern = sentence_delimiters
        self.device = device

    @torch.no_grad()
    def compute_all_levels(
        self,
        model: PreTrainedModel,
        texts: Union[str, List[str]],
        return_details: bool = False
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Compute multi-level perplexities for text(s).

        Args:
            model: HuggingFace model
            texts: Single text or list of texts
            return_details: Whether to return detailed breakdown

        Returns:
            Dictionary with 'word', 'phrase', 'sentence' perplexities
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True
        else:
            squeeze = False

        # Tokenize
        encodings = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # Get model outputs
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits

        # Compute per-token log probs
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].float()

        # Word-level perplexity
        word_ppls = self._compute_word_ppl(shift_logits, shift_labels, shift_mask)

        # Phrase-level perplexity
        phrase_ppls = self._compute_phrase_ppl(shift_logits, shift_labels, shift_mask)

        # Sentence-level perplexity
        sentence_ppls = self._compute_sentence_ppl(shift_logits, shift_labels, shift_mask, texts)

        results = {
            'word': word_ppls,
            'phrase': phrase_ppls,
            'sentence': sentence_ppls,
        }

        if squeeze:
            results = {k: v.item() if v.numel() == 1 else v[0].item() for k, v in results.items()}

        return results

    def _compute_word_ppl(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute word-level (token-level) perplexity."""
        # Compute cross entropy loss per sample
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction='none'
        ).reshape(labels.shape)

        # Mask and average
        masked_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Convert to perplexity
        ppl = torch.exp(masked_loss)

        return ppl

    def _compute_phrase_ppl(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute phrase-level perplexity using n-gram chunks."""
        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask
        token_log_probs = token_log_probs * mask

        batch_size = logits.size(0)
        seq_len = logits.size(1)

        phrase_ppls_list = []

        for n in self.phrase_sizes:
            if seq_len < n:
                continue

            # Compute n-gram perplexities
            chunk_log_probs = []

            for i in range(0, seq_len - n + 1, n):
                chunk_lp = token_log_probs[:, i:i+n].sum(dim=1)
                chunk_m = mask[:, i:i+n].sum(dim=1)

                # Only valid chunks
                valid = chunk_m == n
                if valid.any():
                    # Average per token in chunk
                    avg_lp = chunk_lp[valid] / n
                    chunk_log_probs.append(avg_lp)

            if chunk_log_probs:
                # Average across chunks
                all_chunks = torch.cat(chunk_log_probs)
                phrase_ppl = torch.exp(-all_chunks.mean())
                phrase_ppls_list.append(phrase_ppl)

        # Average across n-gram sizes
        if phrase_ppls_list:
            avg_phrase_ppl = torch.stack(phrase_ppls_list).mean()
        else:
            # Fallback to word-level
            avg_phrase_ppl = self._compute_word_ppl(logits, labels, mask).mean()

        # Return per-sample (approximate)
        return avg_phrase_ppl.unsqueeze(0).expand(batch_size)

    def _compute_sentence_ppl(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Compute sentence-level perplexity."""
        # For simplicity, treat entire sequence as one "sentence"
        # In production, you could split by sentence delimiters

        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask and compute per-sequence average
        masked_log_probs = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Convert to perplexity
        sentence_ppl = torch.exp(-masked_log_probs)

        return sentence_ppl


class MultiLevelPPLRewardFunction:
    """
    TRL-compatible reward function using multi-level perplexity.

    This reward function evaluates generated text at three levels:
    - Word-level: Individual token predictions
    - Phrase-level: Multi-token phrase coherence
    - Sentence-level: Full sequence consistency

    Args:
        baseline_model: Model to compute perplexities (can be same as training model)
        tokenizer: Tokenizer
        baselines: Pre-computed baseline PPLs (dict or path to pickle file)
        weights: Tuple of (word_weight, phrase_weight, sentence_weight)
        reward_thresholds: List of (improvement_pct, reward) tuples
        normalize: Whether to normalize rewards
        device: Device to use
    """

    def __init__(
        self,
        baseline_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        baselines: Optional[Union[Dict, str]] = None,
        weights: Tuple[float, float, float] = (0.2, 0.3, 0.5),
        reward_thresholds: List[Tuple[float, float]] = [
            (0.05, 0.2),
            (0.5, 0.5),
            (1.0, 0.7),
            (2.0, 0.9),
            (3.0, 1.0),
        ],
        normalize: bool = False,
        device: str = 'cuda',
        use_cache: bool = True,
    ):
        self.baseline_model = baseline_model
        self.baseline_model.eval()  # Always in eval mode
        self.tokenizer = tokenizer
        self.device = device
        self.weights = weights
        self.reward_thresholds = sorted(reward_thresholds, key=lambda x: x[0])
        self.normalize = normalize
        self.use_cache = use_cache

        # Load baselines
        if baselines is not None:
            if isinstance(baselines, str):
                with open(baselines, 'rb') as f:
                    self.baselines = pickle.load(f)
            else:
                self.baselines = baselines
        else:
            self.baselines = {}

        # Initialize calculator
        self.calculator = MultiLevelPPLCalculator(
            tokenizer=tokenizer,
            phrase_sizes=[2, 3, 4],
            device=device
        )

        # Cache for computed PPLs
        self.ppl_cache = {} if use_cache else None

        logger.info(f"MultiLevelPPLRewardFunction initialized")
        logger.info(f"  Weights: word={weights[0]}, phrase={weights[1]}, sentence={weights[2]}")
        logger.info(f"  Baselines: {len(self.baselines)} entries")

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute rewards for completions.

        This is the main interface for TRL GRPOTrainer.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings

        Returns:
            Tensor of rewards [batch_size]
        """
        # Combine prompts and completions
        full_texts = [p + c for p, c in zip(prompts, completions)]

        return self.compute_rewards(full_texts, prompts=prompts, completions=completions)

    @torch.no_grad()
    def compute_rewards(
        self,
        texts: List[str],
        prompts: Optional[List[str]] = None,
        completions: Optional[List[str]] = None,
        return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute multi-level PPL rewards.

        Args:
            texts: Full texts to evaluate
            prompts: Optional prompts (for caching/lookup)
            completions: Optional completions (for caching/lookup)
            return_details: Whether to return detailed metrics

        Returns:
            Rewards tensor [batch_size], optionally with details dict
        """
        batch_size = len(texts)

        # Compute current PPLs
        current_ppls = self.calculator.compute_all_levels(
            model=self.baseline_model,
            texts=texts
        )

        # Convert to per-sample if batched
        if not isinstance(current_ppls['word'], (int, float)):
            current_ppls_list = [
                {
                    'word': current_ppls['word'][i].item(),
                    'phrase': current_ppls['phrase'][i].item(),
                    'sentence': current_ppls['sentence'][i].item(),
                }
                for i in range(batch_size)
            ]
        else:
            current_ppls_list = [current_ppls]

        # Compute rewards
        rewards = []
        improvements = {'word': [], 'phrase': [], 'sentence': [], 'avg': []}

        for i, text in enumerate(texts):
            current = current_ppls_list[i] if batch_size > 1 else current_ppls

            # Get baseline (or use current as baseline if not available)
            baseline = self._get_baseline(text, prompts[i] if prompts else None)

            if baseline is None:
                # No baseline - use current as baseline (0 improvement)
                baseline = current.copy()

            # Compute improvements
            word_imp = self._compute_improvement(baseline['word'], current['word'])
            phrase_imp = self._compute_improvement(baseline['phrase'], current['phrase'])
            sentence_imp = self._compute_improvement(baseline['sentence'], current['sentence'])

            # Weighted average
            w_word, w_phrase, w_sentence = self.weights
            total_weight = w_word + w_phrase + w_sentence
            avg_imp = (
                word_imp * w_word / total_weight +
                phrase_imp * w_phrase / total_weight +
                sentence_imp * w_sentence / total_weight
            )

            # Convert to reward
            reward = self._improvement_to_reward(avg_imp)

            rewards.append(reward)
            improvements['word'].append(word_imp)
            improvements['phrase'].append(phrase_imp)
            improvements['sentence'].append(sentence_imp)
            improvements['avg'].append(avg_imp)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        if self.normalize:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        if return_details:
            details = {
                'rewards': rewards_tensor,
                'improvements': {k: torch.tensor(v) for k, v in improvements.items()},
                'current_ppls': current_ppls_list if batch_size > 1 else [current_ppls],
            }
            return rewards_tensor, details

        return rewards_tensor

    def _get_baseline(
        self,
        text: str,
        prompt: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """Get baseline PPLs for text."""
        # Try to find in baselines dict
        if prompt and prompt in self.baselines:
            baseline = self.baselines[prompt]
            return {
                'word': baseline.get('word', baseline.get('baseline_ppl', 10.0)),
                'phrase': baseline.get('phrase', baseline.get('baseline_ppl', 10.0) * 1.1),
                'sentence': baseline.get('sentence', baseline.get('baseline_ppl', 10.0) * 1.2),
            }

        # Try by text
        if text in self.baselines:
            baseline = self.baselines[text]
            return {
                'word': baseline.get('word', baseline.get('baseline_ppl', 10.0)),
                'phrase': baseline.get('phrase', baseline.get('baseline_ppl', 10.0) * 1.1),
                'sentence': baseline.get('sentence', baseline.get('baseline_ppl', 10.0) * 1.2),
            }

        return None

    def _compute_improvement(self, baseline: float, current: float) -> float:
        """Compute percentage improvement."""
        if baseline <= 0 or baseline == float('inf'):
            return 0.0

        improvement = (baseline - current) / baseline * 100
        return improvement

    def _improvement_to_reward(self, improvement: float) -> float:
        """Convert improvement percentage to reward."""
        reward = 0.0

        for threshold, reward_value in self.reward_thresholds:
            if improvement >= threshold:
                reward = reward_value

        return max(reward, 0.0)

    def add_baseline(self, key: str, baseline_ppls: Dict[str, float]):
        """Add a baseline entry."""
        self.baselines[key] = baseline_ppls

    def add_baselines_batch(self, baselines_dict: Dict[str, Dict[str, float]]):
        """Add multiple baseline entries."""
        self.baselines.update(baselines_dict)


def create_reward_function(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    baselines_path: Optional[str] = None,
    **kwargs
) -> MultiLevelPPLRewardFunction:
    """
    Convenience function to create reward function.

    Args:
        model: Model to use for PPL computation
        tokenizer: Tokenizer
        baselines_path: Path to baselines pickle file
        **kwargs: Additional arguments for MultiLevelPPLRewardFunction

    Returns:
        Configured reward function
    """
    return MultiLevelPPLRewardFunction(
        baseline_model=model,
        tokenizer=tokenizer,
        baselines=baselines_path,
        **kwargs
    )


# Alias for backward compatibility
create_multilevel_reward_function = create_reward_function
