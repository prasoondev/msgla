from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from fla.layers.gla import GatedLinearAttention
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.models.utils import Cache

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


def _compute_branch_expand_ratios(
    hidden_size: int,
    expand_k: float,
    expand_v: float,
    total_heads: int,
    branch_heads: int,
) -> tuple[float, float]:
    total_key_dim = int(hidden_size * expand_k)
    total_value_dim = int(hidden_size * expand_v)
    if total_key_dim % total_heads != 0:
        raise ValueError("`hidden_size * expand_k` must be divisible by `num_heads`.")
    if total_value_dim % total_heads != 0:
        raise ValueError("`hidden_size * expand_v` must be divisible by `num_heads`.")

    head_k_dim = total_key_dim // total_heads
    head_v_dim = total_value_dim // total_heads
    branch_key_dim = head_k_dim * branch_heads
    branch_value_dim = head_v_dim * branch_heads
    return branch_key_dim / hidden_size, branch_value_dim / hidden_size


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if scale == 1:
        return hidden_states, attention_mask

    batch_size, seq_len, hidden_size = hidden_states.shape
    pad_len = (-seq_len) % scale

    if pad_len > 0:
        hidden_states = torch.cat(
            [hidden_states, hidden_states.new_zeros(batch_size, pad_len, hidden_size)],
            dim=1,
        )
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_zeros(batch_size, pad_len)],
                dim=1,
            )

    pooled_len = hidden_states.shape[1] // scale
    hidden_states = hidden_states.reshape(batch_size, pooled_len, scale, hidden_size)

    if attention_mask is None:
        pooled_states = hidden_states.mean(dim=2)
        pooled_mask = None
    else:
        mask = attention_mask.reshape(batch_size, pooled_len, scale).to(hidden_states.dtype)
        denom = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        pooled_states = (hidden_states * mask.unsqueeze(-1)).sum(dim=2) / denom
        pooled_mask = (mask.sum(dim=2) > 0).to(attention_mask.dtype)

    return pooled_states, pooled_mask


def _upsample_with_causal_hold(
    branch_output: torch.Tensor,
    scale: int,
    target_len: int,
) -> torch.Tensor:
    if scale == 1:
        return branch_output[:, :target_len]

    batch_size, _, hidden_size = branch_output.shape
    repeated = branch_output.repeat_interleave(scale, dim=1)
    aligned = branch_output.new_zeros(batch_size, repeated.shape[1] + scale - 1, hidden_size)
    aligned[:, scale - 1:scale - 1 + repeated.shape[1]] = repeated
    return aligned[:, :target_len]


@dataclass
class MSGLABranchDecodeState:
    past_key_values: Cache
    pending_hidden_states: torch.Tensor | None
    last_output: torch.Tensor | None


@dataclass
class MSGLADecodeState:
    branches: list[list[MSGLABranchDecodeState]]


def _ensure_cache_layer(cache: Cache, layer_idx: int = 0) -> None:
    if getattr(cache, "use_layer_class_to_replicate", False):
        while len(cache.layers) <= layer_idx:
            cache.layers.append(cache.layer_class_to_replicate())
    else:
        cache.append_new_layers(layer_idx)


def _concat_cache_items(items):
    non_none = [item for item in items if item is not None]
    if not non_none:
        return None
    first = non_none[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(items, dim=0)
    if isinstance(first, tuple):
        return tuple(_concat_cache_items([item[i] for item in items]) for i in range(len(first)))
    if isinstance(first, list):
        return [_concat_cache_items([item[i] for item in items]) for i in range(len(first))]
    raise TypeError(f"Unsupported cache item type `{type(first).__name__}` for batched decode.")


def _slice_cache_item(item, idx: int):
    if item is None:
        return None
    if isinstance(item, torch.Tensor):
        return item[idx:idx + 1].contiguous()
    if isinstance(item, tuple):
        return tuple(_slice_cache_item(x, idx) for x in item)
    if isinstance(item, list):
        return [_slice_cache_item(x, idx) for x in item]
    raise TypeError(f"Unsupported cache item type `{type(item).__name__}` for batched decode.")


def _pending_length(pending_hidden_states: torch.Tensor | None) -> int:
    return 0 if pending_hidden_states is None else int(pending_hidden_states.shape[0])


class MultiScaleGatedLinearAttention(nn.Module):
    """
    A training-first multiscale wrapper around GLA.

    Each scale owns a smaller GLA branch whose head budget is a partition of the
    baseline model's total heads. Coarser branches operate on pooled tokens and
    their outputs are causally held until the next pooled update.
    """

    def __init__(
        self,
        mode: str = "chunk",
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        scales: list[int] | tuple[int, ...] = (1, 2, 4),
        scale_num_heads: list[int] | tuple[int, ...] = (2, 1, 1),
        feature_map: str | None = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_output_gate: bool = True,
        gate_fn: str = "swish",
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        clamp_min: float | None = None,
        fuse_norm: bool = True,
        pool_mode: str = "avg",
        fuse_mode: str = "softmax",
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.scales = list(scales)
        self.scale_num_heads = list(scale_num_heads)
        self.pool_mode = pool_mode
        self.fuse_mode = fuse_mode
        self.layer_idx = layer_idx

        if len(self.scales) != len(self.scale_num_heads):
            raise ValueError("`scales` and `scale_num_heads` must have the same length.")

        self.branches = nn.ModuleList()
        for scale, branch_heads in zip(self.scales, self.scale_num_heads):
            branch_expand_k, branch_expand_v = _compute_branch_expand_ratios(
                hidden_size=hidden_size,
                expand_k=expand_k,
                expand_v=expand_v,
                total_heads=num_heads,
                branch_heads=branch_heads,
            )
            self.branches.append(
                GatedLinearAttention(
                    mode=mode,
                    hidden_size=hidden_size,
                    expand_k=branch_expand_k,
                    expand_v=branch_expand_v,
                    num_heads=branch_heads,
                    num_kv_heads=branch_heads,
                    feature_map=feature_map,
                    use_short_conv=use_short_conv,
                    conv_size=conv_size,
                    use_output_gate=use_output_gate,
                    gate_fn=gate_fn,
                    elementwise_affine=elementwise_affine,
                    norm_eps=norm_eps,
                    clamp_min=clamp_min,
                    fuse_norm=fuse_norm,
                    layer_idx=0,
                )
            )

        self.fuse = nn.Linear(hidden_size, len(self.scales), bias=True)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.fuse.bias.data[0] = 1.0

    def _build_empty_decode_state(
        self,
        batch_size: int,
        hidden_states: torch.Tensor,
    ) -> MSGLADecodeState:
        branches = []
        for _ in self.scales:
            branch_states = []
            for _ in range(batch_size):
                branch_states.append(
                    MSGLABranchDecodeState(
                        past_key_values=Cache(),
                        pending_hidden_states=None,
                        last_output=hidden_states.new_zeros(self.hidden_size),
                    )
                )
            branches.append(branch_states)
        return MSGLADecodeState(branches=branches)

    def _get_decode_state(
        self,
        batch_size: int,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
    ) -> tuple[MSGLADecodeState, bool]:
        last_state = get_layer_cache(self, past_key_values)
        if last_state is None or last_state["recurrent_state"] is None:
            return self._build_empty_decode_state(batch_size, hidden_states), False
        decode_state = last_state["recurrent_state"]
        if not isinstance(decode_state, MSGLADecodeState):
            raise TypeError(f"Expected `MSGLADecodeState`, got `{type(decode_state).__name__}`.")
        if len(decode_state.branches) != len(self.scales):
            raise ValueError("Cached MS-GLA state has a different number of scales than the current layer.")
        return decode_state, True

    def _decode_branch_token(
        self,
        branch: GatedLinearAttention,
        branch_state: MSGLABranchDecodeState,
        hidden_token: torch.Tensor,
        token_is_valid: bool,
        scale: int,
    ) -> torch.Tensor:
        if not token_is_valid:
            return hidden_token.new_zeros(self.hidden_size)

        pending = branch_state.pending_hidden_states
        token = hidden_token.unsqueeze(0)
        if pending is None:
            pending = token
        else:
            pending = torch.cat([pending, token], dim=0)

        if pending.shape[0] == scale:
            pooled = pending.mean(dim=0, keepdim=True).unsqueeze(0)
            branch_out, _, branch_cache = branch(
                hidden_states=pooled,
                attention_mask=None,
                past_key_values=branch_state.past_key_values,
                use_cache=True,
                output_attentions=False,
            )
            branch_state.past_key_values = branch_cache
            branch_state.last_output = branch_out[0, 0]
            pending = None

        branch_state.pending_hidden_states = pending
        if branch_state.last_output is None:
            return hidden_token.new_zeros(self.hidden_size)
        return branch_state.last_output

    def _merge_branch_caches(
        self,
        branch_states: list[MSGLABranchDecodeState],
    ) -> Cache:
        merged_cache = Cache(seen_tokens=branch_states[0].past_key_values.get_seq_length(0))
        cache_has_layers = [len(branch_state.past_key_values) > 0 for branch_state in branch_states]
        if any(cache_has_layers) and not all(cache_has_layers):
            raise ValueError("Cannot batch decode with partially initialized branch caches.")
        if not any(cache_has_layers):
            return merged_cache

        merged_state = {}
        for key in ("recurrent_state", "attn_state", "conv_state", "ffn_state"):
            items = [branch_state.past_key_values[0][key] for branch_state in branch_states]
            if any(item is None for item in items) and not all(item is None for item in items):
                raise ValueError(f"Cannot batch decode with partially initialized `{key}` cache state.")
            merged_state[key] = _concat_cache_items(items)

        _ensure_cache_layer(merged_cache)
        merged_cache.layers[0].state = merged_state
        merged_cache.layers[0]._seen_tokens = branch_states[0].past_key_values.get_seq_length(0)
        merged_cache._seen_tokens = branch_states[0].past_key_values.get_seq_length(0)
        return merged_cache

    def _can_batch_branch_caches(
        self,
        branch_states: list[MSGLABranchDecodeState],
    ) -> bool:
        cache_has_layers = [len(branch_state.past_key_values) > 0 for branch_state in branch_states]
        if any(cache_has_layers) and not all(cache_has_layers):
            return False
        if not any(cache_has_layers):
            return True

        cache_lengths = {branch_state.past_key_values.get_seq_length(0) for branch_state in branch_states}
        if len(cache_lengths) != 1:
            return False

        for key in ("recurrent_state", "attn_state", "conv_state", "ffn_state"):
            items = [branch_state.past_key_values[0][key] for branch_state in branch_states]
            if any(item is None for item in items) and not all(item is None for item in items):
                return False
        return True

    def _scatter_branch_cache(
        self,
        merged_cache: Cache,
        branch_states: list[MSGLABranchDecodeState],
    ) -> None:
        if len(merged_cache) == 0:
            return
        merged_state = merged_cache[0]
        seen_tokens = merged_cache.get_seq_length(0)
        for batch_idx, branch_state in enumerate(branch_states):
            cache = branch_state.past_key_values
            _ensure_cache_layer(cache)
            cache.layers[0].state = {
                key: _slice_cache_item(value, batch_idx)
                for key, value in merged_state.items()
            }
            cache.layers[0]._seen_tokens = seen_tokens
            cache._seen_tokens = seen_tokens

    def _try_batched_decode_branch(
        self,
        branch: GatedLinearAttention,
        branch_states: list[MSGLABranchDecodeState],
        hidden_token: torch.Tensor,
        scale: int,
    ) -> torch.Tensor | None:
        pending_lengths = {_pending_length(branch_state.pending_hidden_states) for branch_state in branch_states}
        if len(pending_lengths) != 1:
            return None

        pending_len = pending_lengths.pop()
        if pending_len + 1 == scale:
            if not self._can_batch_branch_caches(branch_states):
                return None
            pooled_rows = []
            for batch_idx, branch_state in enumerate(branch_states):
                token = hidden_token[batch_idx].unsqueeze(0)
                if branch_state.pending_hidden_states is None:
                    pooled_rows.append(token[0])
                else:
                    pooled_rows.append(torch.cat([branch_state.pending_hidden_states, token], dim=0).mean(dim=0))
            pooled = torch.stack(pooled_rows, dim=0).unsqueeze(1)
            merged_cache = self._merge_branch_caches(branch_states)
            branch_output, _, merged_cache = branch(
                hidden_states=pooled,
                attention_mask=None,
                past_key_values=merged_cache,
                use_cache=True,
                output_attentions=False,
            )
            self._scatter_branch_cache(merged_cache, branch_states)
            outputs = branch_output[:, 0]
            for batch_idx, branch_state in enumerate(branch_states):
                branch_state.last_output = outputs[batch_idx]
                branch_state.pending_hidden_states = None
            return outputs

        for batch_idx, branch_state in enumerate(branch_states):
            token = hidden_token[batch_idx].unsqueeze(0)
            if branch_state.pending_hidden_states is None:
                branch_state.pending_hidden_states = token
            else:
                branch_state.pending_hidden_states = torch.cat([branch_state.pending_hidden_states, token], dim=0)
            if branch_state.last_output is None:
                branch_state.last_output = hidden_token.new_zeros(self.hidden_size)
        return torch.stack([branch_state.last_output for branch_state in branch_states], dim=0)

    def _forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> tuple[torch.Tensor, Cache | None]:
        batch_size, seq_len, _ = hidden_states.shape
        current_mask = (
            attention_mask[:, -seq_len:].to(torch.bool)
            if attention_mask is not None
            else torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        )

        decode_state, has_existing_state = self._get_decode_state(batch_size, hidden_states, past_key_values)
        if not has_existing_state and seq_len > 1:
            output = self._prefill_with_cache(hidden_states, current_mask, decode_state)
            update_layer_cache(self, past_key_values, recurrent_state=decode_state, offset=seq_len)
            return output, past_key_values

        output = self._decode_tokens_with_cache(hidden_states, current_mask, decode_state)
        update_layer_cache(self, past_key_values, recurrent_state=decode_state, offset=seq_len)
        return output, past_key_values

    def _prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        current_mask: torch.Tensor,
        decode_state: MSGLADecodeState,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        branch_outputs = []

        for branch_idx, (scale, branch) in enumerate(zip(self.scales, self.branches)):
            branch_batch_output = hidden_states.new_zeros(batch_size, seq_len, self.hidden_size)
            for batch_idx in range(batch_size):
                valid_positions = current_mask[batch_idx].nonzero(as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue

                branch_state = decode_state.branches[branch_idx][batch_idx]
                valid_hidden = hidden_states[batch_idx, valid_positions]
                num_valid = valid_hidden.shape[0]
                full_groups = num_valid // scale
                consumed = full_groups * scale
                valid_output = hidden_states.new_zeros(num_valid, self.hidden_size)

                if full_groups > 0:
                    pooled = valid_hidden[:consumed].reshape(1, full_groups, scale, self.hidden_size).mean(dim=2)
                    branch_output, _, branch_cache = branch(
                        hidden_states=pooled,
                        attention_mask=None,
                        past_key_values=branch_state.past_key_values,
                        use_cache=True,
                        output_attentions=False,
                    )
                    branch_state.past_key_values = branch_cache
                    branch_state.last_output = branch_output[0, -1]
                    valid_output[:consumed] = _upsample_with_causal_hold(branch_output, scale, consumed)[0]

                if consumed < num_valid:
                    branch_state.pending_hidden_states = valid_hidden[consumed:].clone()
                    hold_value = branch_state.last_output
                    if hold_value is None:
                        hold_value = hidden_states.new_zeros(self.hidden_size)
                        branch_state.last_output = hold_value
                    valid_output[consumed:] = hold_value.unsqueeze(0).expand(num_valid - consumed, -1)
                else:
                    branch_state.pending_hidden_states = None
                    if branch_state.last_output is None:
                        branch_state.last_output = hidden_states.new_zeros(self.hidden_size)

                branch_batch_output[batch_idx, valid_positions] = valid_output

            branch_outputs.append(branch_batch_output)

        stacked = torch.stack(branch_outputs, dim=2)
        if self.fuse_mode == "softmax":
            branch_weights = torch.softmax(self.fuse(hidden_states), dim=-1)
            output = (stacked * branch_weights.unsqueeze(-1)).sum(dim=2)
        else:
            output = stacked.mean(dim=2)
        return output * current_mask.to(output.dtype).unsqueeze(-1)

    def _decode_tokens_with_cache(
        self,
        hidden_states: torch.Tensor,
        current_mask: torch.Tensor,
        decode_state: MSGLADecodeState,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        outputs = []
        for token_idx in range(hidden_states.shape[1]):
            token_hidden = hidden_states[:, token_idx]
            token_is_all_valid = bool(current_mask[:, token_idx].all().item())
            can_try_batched_decode = batch_size > 1 and token_is_all_valid
            token_outputs = []
            for branch_idx, (scale, branch) in enumerate(zip(self.scales, self.branches)):
                branch_states = decode_state.branches[branch_idx]
                if can_try_batched_decode:
                    branch_output = self._try_batched_decode_branch(branch, branch_states, token_hidden, scale)
                    if branch_output is not None:
                        token_outputs.append(branch_output)
                        continue

                branch_rows = []
                for batch_idx in range(batch_size):
                    branch_rows.append(
                        self._decode_branch_token(
                            branch=branch,
                            branch_state=branch_states[batch_idx],
                            hidden_token=token_hidden[batch_idx],
                            token_is_valid=bool(current_mask[batch_idx, token_idx].item()),
                            scale=scale,
                        )
                    )
                token_outputs.append(torch.stack(branch_rows, dim=0))

            stacked = torch.stack(token_outputs, dim=1)
            if self.fuse_mode == "softmax":
                branch_weights = torch.softmax(self.fuse(token_hidden), dim=-1)
                fused = (stacked * branch_weights.unsqueeze(-1)).sum(dim=1)
            else:
                fused = stacked.mean(dim=1)
            fused = fused * current_mask[:, token_idx].to(fused.dtype).unsqueeze(-1)
            outputs.append(fused)

        return torch.stack(outputs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        if output_attentions:
            output_attentions = False

        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("MS-GLA expects a padding mask of shape [batch_size, seq_len].")

        if use_cache:
            output, past_key_values = self._forward_with_cache(hidden_states, attention_mask, past_key_values)
            return output, None, past_key_values
        if past_key_values is not None:
            raise NotImplementedError("Passing `past_key_values` requires `use_cache=True` for MS-GLA.")

        seq_len = hidden_states.shape[1]
        branch_outputs = []
        for scale, branch in zip(self.scales, self.branches):
            pooled_states, pooled_mask = _pool_hidden_states(hidden_states, attention_mask, scale)
            branch_output, _, _ = branch(
                hidden_states=pooled_states,
                attention_mask=pooled_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                **kwargs,
            )
            branch_outputs.append(_upsample_with_causal_hold(branch_output, scale, seq_len))

        stacked = torch.stack(branch_outputs, dim=2)
        if self.fuse_mode == "softmax":
            branch_weights = torch.softmax(self.fuse(hidden_states), dim=-1)
            output = (stacked * branch_weights.unsqueeze(-1)).sum(dim=2)
        else:
            output = stacked.mean(dim=2)

        if attention_mask is not None:
            output = output * attention_mask.to(output.dtype).unsqueeze(-1)

        return output, None, None
