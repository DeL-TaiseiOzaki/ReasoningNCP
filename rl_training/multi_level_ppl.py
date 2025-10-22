"""
Multi-Level Perplexity Calculation Module

This module implements three levels of perplexity evaluation:
1. Word-level: Individual token prediction quality
2. Phrase-level: Multi-token phrase coherence
3. Sentence-level: Full utterance consistency

References:
- Perplexity as a language model evaluation metric
- https://www.baeldung.com/cs/nlp-perplexity
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import re


class MultiLevelPerplexityCalculator:
    """
    Calculate perplexity at multiple granularity levels.

    Args:
        tokenizer: HuggingFace tokenizer for text processing
        phrase_sizes: List of n-gram sizes for phrase-level evaluation (default: [2, 3, 4])
        sentence_delimiters: Regex pattern for sentence splitting
    """

    def __init__(
        self,
        tokenizer,
        phrase_sizes: List[int] = [2, 3, 4],
        sentence_delimiters: str = r'[.!?。！？]'
    ):
        self.tokenizer = tokenizer
        self.phrase_sizes = phrase_sizes
        self.sentence_delimiter_pattern = sentence_delimiters

    @torch.no_grad()
    def compute_word_level_ppl(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute word-level (token-level) perplexity.

        This measures how well the model predicts individual tokens.
        Lower perplexity indicates better token prediction.

        Args:
            model: Language model
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            labels: Target labels (use -100 for tokens to ignore)

        Returns:
            perplexity (float): Overall word-level perplexity
            details (dict): Detailed metrics including loss and token count
        """
        # Forward pass
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # Get loss (already averaged over valid tokens by HuggingFace)
        loss = outputs.loss

        # Calculate perplexity: PPL = exp(loss)
        perplexity = torch.exp(loss).item()

        # Count valid tokens (not -100)
        valid_tokens = (labels != -100).sum().item()

        details = {
            'loss': loss.item(),
            'perplexity': perplexity,
            'valid_tokens': valid_tokens,
            'level': 'word'
        }

        return perplexity, details

    @torch.no_grad()
    def compute_phrase_level_ppl(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute phrase-level perplexity using n-gram chunks.

        This evaluates how well the model predicts meaningful multi-token phrases.
        We compute perplexity for different n-gram sizes and average them.

        Args:
            model: Language model
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            labels: Target labels

        Returns:
            perplexity (float): Average phrase-level perplexity
            details (dict): Per-phrase-size metrics
        """
        # Get logits
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Calculate per-token log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probabilities for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]

        # Mask out ignored tokens (-100)
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask

        # Compute phrase-level perplexities for different n-gram sizes
        phrase_ppls = []
        phrase_details = {}

        for n in self.phrase_sizes:
            if shift_labels.size(1) < n:
                continue

            # Split sequence into n-gram chunks
            chunk_log_probs = []
            chunk_masks = []

            for i in range(0, shift_labels.size(1) - n + 1, n):
                chunk_lp = token_log_probs[:, i:i+n].sum(dim=1)
                chunk_m = mask[:, i:i+n].sum(dim=1)

                # Only consider chunks with all valid tokens
                valid_chunks = chunk_m == n
                if valid_chunks.any():
                    chunk_log_probs.append(chunk_lp[valid_chunks])
                    chunk_masks.append(chunk_m[valid_chunks])

            if len(chunk_log_probs) > 0:
                # Average log probability across chunks
                all_chunk_lps = torch.cat(chunk_log_probs)
                avg_chunk_lp = all_chunk_lps.mean()

                # Normalize by chunk size and convert to perplexity
                avg_lp_per_token = avg_chunk_lp / n
                phrase_ppl = torch.exp(-avg_lp_per_token).item()

                phrase_ppls.append(phrase_ppl)
                phrase_details[f'phrase_{n}gram'] = {
                    'perplexity': phrase_ppl,
                    'num_chunks': len(all_chunk_lps)
                }

        # Average across different phrase sizes
        if len(phrase_ppls) > 0:
            avg_phrase_ppl = sum(phrase_ppls) / len(phrase_ppls)
        else:
            # Fallback to word-level if sequence too short
            avg_phrase_ppl = torch.exp(-token_log_probs.sum() / mask.sum()).item()

        details = {
            'perplexity': avg_phrase_ppl,
            'phrase_sizes': self.phrase_sizes,
            'phrase_details': phrase_details,
            'level': 'phrase'
        }

        return avg_phrase_ppl, details

    @torch.no_grad()
    def compute_sentence_level_ppl(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        text: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute sentence/utterance-level perplexity.

        This evaluates coherence over complete sentences or utterances.
        If text is provided, we split into sentences; otherwise treat entire
        sequence as one utterance.

        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            text: Optional decoded text for sentence splitting

        Returns:
            perplexity (float): Sentence-level perplexity
            details (dict): Per-sentence metrics
        """
        # Get logits
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Calculate per-token log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask

        # If text provided, split into sentences
        if text is not None:
            sentences = self._split_into_sentences(text)
            if len(sentences) > 1:
                return self._compute_per_sentence_ppl(
                    token_log_probs, mask, shift_labels, sentences
                )

        # Otherwise, treat entire sequence as one utterance
        total_log_prob = token_log_probs.sum()
        total_tokens = mask.sum()

        if total_tokens > 0:
            avg_log_prob = total_log_prob / total_tokens
            sentence_ppl = torch.exp(-avg_log_prob).item()
        else:
            sentence_ppl = float('inf')

        details = {
            'perplexity': sentence_ppl,
            'total_tokens': total_tokens.item(),
            'num_sentences': 1,
            'level': 'sentence'
        }

        return sentence_ppl, details

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using delimiter pattern."""
        sentences = re.split(self.sentence_delimiter_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_per_sentence_ppl(
        self,
        token_log_probs: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        sentences: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute perplexity per sentence and average."""
        # This is a simplified version - ideally we'd align tokens to sentences
        # For now, we split the sequence into equal parts
        seq_len = token_log_probs.size(1)
        num_sentences = len(sentences)

        sentence_ppls = []
        chunk_size = seq_len // num_sentences

        for i in range(num_sentences):
            start_idx = i * chunk_size
            end_idx = seq_len if i == num_sentences - 1 else (i + 1) * chunk_size

            chunk_lp = token_log_probs[:, start_idx:end_idx].sum()
            chunk_tokens = mask[:, start_idx:end_idx].sum()

            if chunk_tokens > 0:
                avg_lp = chunk_lp / chunk_tokens
                sent_ppl = torch.exp(-avg_lp).item()
                sentence_ppls.append(sent_ppl)

        avg_sentence_ppl = sum(sentence_ppls) / len(sentence_ppls) if sentence_ppls else float('inf')

        details = {
            'perplexity': avg_sentence_ppl,
            'num_sentences': num_sentences,
            'sentence_ppls': sentence_ppls,
            'level': 'sentence'
        }

        return avg_sentence_ppl, details

    @torch.no_grad()
    def compute_all_levels(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        text: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Compute perplexity at all three levels.

        Returns:
            Dictionary with 'word', 'phrase', 'sentence' perplexities and details
        """
        word_ppl, word_details = self.compute_word_level_ppl(
            model, input_ids, attention_mask, labels
        )

        phrase_ppl, phrase_details = self.compute_phrase_level_ppl(
            model, input_ids, attention_mask, labels
        )

        sentence_ppl, sentence_details = self.compute_sentence_level_ppl(
            model, input_ids, attention_mask, labels, text
        )

        return {
            'word': {'perplexity': word_ppl, 'details': word_details},
            'phrase': {'perplexity': phrase_ppl, 'details': phrase_details},
            'sentence': {'perplexity': sentence_ppl, 'details': sentence_details},
        }


def aggregate_multi_level_ppl(
    word_ppl: float,
    phrase_ppl: float,
    sentence_ppl: float,
    weights: Tuple[float, float, float] = (0.3, 0.3, 0.4)
) -> float:
    """
    Aggregate multi-level perplexities with weights.

    Args:
        word_ppl: Word-level perplexity
        phrase_ppl: Phrase-level perplexity
        sentence_ppl: Sentence-level perplexity
        weights: Tuple of (word_weight, phrase_weight, sentence_weight)
                 Default gives more weight to sentence-level coherence

    Returns:
        Weighted average perplexity
    """
    w_word, w_phrase, w_sentence = weights

    # Normalize weights
    total_weight = w_word + w_phrase + w_sentence
    w_word /= total_weight
    w_phrase /= total_weight
    w_sentence /= total_weight

    # Weighted average (in log space for numerical stability)
    import math
    log_ppl = (
        w_word * math.log(word_ppl) +
        w_phrase * math.log(phrase_ppl) +
        w_sentence * math.log(sentence_ppl)
    )

    return math.exp(log_ppl)


def compute_multi_level_reward(
    baseline_ppls: Dict[str, float],
    current_ppls: Dict[str, float],
    weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
    reward_thresholds: List[Tuple[float, float]] = [
        (0.05, 0.3),   # >= 0.05% improvement -> 0.3 reward
        (0.5, 0.5),    # >= 0.5% improvement -> 0.5 reward
        (1.0, 0.7),    # >= 1% improvement -> 0.7 reward
        (2.0, 1.0),    # >= 2% improvement -> 1.0 reward
    ]
) -> Tuple[float, Dict[str, any]]:
    """
    Compute reward based on multi-level perplexity improvements.

    Args:
        baseline_ppls: Dict with 'word', 'phrase', 'sentence' baseline perplexities
        current_ppls: Dict with 'word', 'phrase', 'sentence' current perplexities
        weights: Importance weights for each level
        reward_thresholds: List of (improvement_percentage, reward) tuples

    Returns:
        reward (float): Computed reward value
        info (dict): Detailed information about improvements
    """
    # Calculate percentage improvements for each level
    improvements = {}
    for level in ['word', 'phrase', 'sentence']:
        baseline = baseline_ppls.get(level, float('inf'))
        current = current_ppls.get(level, float('inf'))

        if baseline > 0 and baseline != float('inf'):
            improvement = (baseline - current) / baseline * 100
        else:
            improvement = 0.0

        improvements[level] = improvement

    # Weighted average improvement
    w_word, w_phrase, w_sentence = weights
    total_weight = w_word + w_phrase + w_sentence

    avg_improvement = (
        improvements['word'] * w_word / total_weight +
        improvements['phrase'] * w_phrase / total_weight +
        improvements['sentence'] * w_sentence / total_weight
    )

    # Determine reward based on thresholds
    reward = 0.0
    for threshold, reward_value in sorted(reward_thresholds, key=lambda x: x[0]):
        if avg_improvement >= threshold:
            reward = reward_value

    # Ensure non-negative
    reward = max(reward, 0.0)

    info = {
        'improvements': improvements,
        'avg_improvement': avg_improvement,
        'reward': reward,
        'baseline_ppls': baseline_ppls,
        'current_ppls': current_ppls,
    }

    return reward, info
