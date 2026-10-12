# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused sliding-window attention + ring buffer KV cache update composite.

`odml.fused_sdpa_cache_update` attends over the ring buffer *before* this
step's write plus the new tokens, and optionally writes the new tokens into the
ring buffer. Doing both in one op lets the backend order the write after every
read of the old cache even though the runtimes alias the cache input and
output buffers.

Signature (T = new tokens, W = ring buffer size, G = query heads per KV head):
  inputs:
    0 query        [1, Hkv, G*T, D]  pre-scaled, heads packed g-major.
    1 key_cache    [1, Hkv, W, D]
    2 value_cache  [1, Hkv, D, W]
    3 key_new      [1, Hkv, T, D]
    4 value_new    [1, Hkv, D, T]
    5 mask         [1, 1, T, W + T]  bool (True = attend) or additive float.
                   Columns [0, W) are the cache slots before the write.
    6 param        int32 runtime param tensor, [0] = start, [1] = end; the
                   number of valid new tokens is `end - start`. Must be the
                   signature input so every backend sees its runtime value.
  outputs:
    0 attention    [1, Hkv, G*T, D]
    1 key_cache'   only if `update_cache`
    2 value_cache' only if `update_cache`
  attributes: cache_size (W), update_cache, softcap (only when set).

The cache write puts new token i into slot (start + i) % W for the last
min(valid, W) valid tokens.
"""

from litert_torch.backend import composite
from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
from litert_torch.generative.export_hf.experimental.composites import cache_update as gpu_cache_update
import torch
import torch.nn.functional as F

COMPOSITE_NAME = "odml.fused_sdpa_cache_update"
# Matches sdpa.MASK_FILL_VALUE (ML Drift convention).
_MASK_FILL_VALUE = -10000.0


def _repeat_rows(mask: torch.Tensor, g: int) -> torch.Tensor:
  """Repeats mask rows for `g` packed query heads ([1,1,T,S] -> [1,1,gT,S])."""
  if g == 1:
    return mask
  if mask.dtype == torch.bool:
    # Bool concatenation is routed through float32, which is what the export
    # pipeline supports (see sdpa._broadcast_mask_to_packed_query).
    return torch.cat([mask.to(torch.float32)] * g, dim=-2) != 0
  return torch.cat([mask] * g, dim=-2)


def _ring_write(
    cache: torch.Tensor,
    update: torch.Tensor,
    start: torch.Tensor,
    num_valid: torch.Tensor,
    valid_dep: torch.Tensor,
    ts_idx: int,
) -> torch.Tensor:
  """Writes the last min(num_valid, W) valid tokens of `update` into `cache`."""
  cache_size = cache.size(ts_idx)
  new_len = update.size(ts_idx)
  idx = torch.arange(new_len, dtype=torch.int32, device=update.device)
  # Keeping only the last W valid tokens gives every slot at most one writer,
  # so the one-hot routing does not sum tokens that wrap onto the same slot.
  keep = (idx < num_valid) & (idx >= num_valid - cache_size) & valid_dep
  return gpu_cache_update.update_kv_cache_with_sliding(
      cache, update, start + idx, keep, ts_idx=ts_idx
  )


def fused_sdpa_cache_update(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_new: torch.Tensor,
    value_new: torch.Tensor,
    mask: torch.Tensor,
    param_tensor: torch.Tensor,
    update_cache: bool,
    softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
  """Emits `odml.fused_sdpa_cache_update`; see the module docstring.

  Args:
    query: [1, Hkv, G*T, D], already multiplied by the attention scale.
    key_cache: [1, Hkv, W, D] ring buffer.
    value_cache: [1, Hkv, D, W] ring buffer.
    key_new: [1, Hkv, T, D].
    value_new: [1, Hkv, D, T].
    mask: [1, 1, T, W + T], bool or additive float.
    param_tensor: The raw runtime param signature input.
    update_cache: Whether the op writes the ring buffer.
    softcap: Optional logit soft-capping value.

  Returns:
    (attention, new_key_cache, new_value_cache); the caches are None when
    `update_cache` is False.
  """
  cache_size = key_cache.size(2)
  new_len = key_new.size(2)
  assert value_cache.size(3) == cache_size, "value cache must be [1,H,D,W]"
  assert value_new.size(3) == new_len, "value_new must be [1,H,D,T]"
  assert mask.size(-1) == cache_size + new_len, "mask must be [1,1,T,W+T]"

  attrs = {"cache_size": cache_size, "update_cache": update_cache}
  if softcap is not None:
    attrs["softcap"] = float(softcap)
  builder = composite.StableHLOCompositeBuilder(
      name=COMPOSITE_NAME, attr=attrs  # pyrefly: ignore[bad-argument-type]
  )
  (
      query,
      key_cache,
      value_cache,
      key_new,
      value_new,
      mask,
      param_tensor,
  ) = builder.mark_inputs(
      query, key_cache, value_cache, key_new, value_new, mask, param_tensor
  )

  g = query.size(2) // new_len
  # `bmm_4d` lowers to a batched `dot_general` (TFLite BATCH_MATMUL), like the
  # unfused SDPA decomposition. `torch.matmul` against the batch-1 caches
  # lowers to a bias-less FULLY_CONNECTED instead; TFLite's composite inlining
  # (used when a CPU delegate does not claim this op) remaps its optional
  # bias index as a real tensor and the decomposition then fails at runtime.
  logits_past = bmm_lib.bmm_4d(query, key_cache)
  logits_new = bmm_lib.bmm_4d(query, key_new)
  logits = torch.cat([logits_past, logits_new], dim=-1)
  if softcap is not None:
    logits = torch.tanh(logits / softcap) * softcap
  full_mask = _repeat_rows(mask, g)
  if full_mask.dtype == torch.bool:
    logits = torch.where(
        full_mask, logits, torch.tensor(_MASK_FILL_VALUE, dtype=logits.dtype)
    )
  else:
    logits = logits + full_mask
  probs = F.softmax(logits, dim=-1).type_as(query)
  probs_past, probs_new = probs.split([cache_size, new_len], dim=-1)
  out_past = bmm_lib.bmm_4d(probs_past, value_cache)
  attention = out_past + bmm_lib.bmm_4d(probs_new, value_new)

  if not update_cache:
    # With a single output, mark_outputs returns the tensor itself.
    attention = builder.mark_outputs(attention)
    return attention, None, None

  flat_param = param_tensor.reshape(-1)
  start = flat_param[0]
  num_valid = torch.clamp(flat_param[1] - start, min=0, max=new_len)
  # Force a data dependency from `out_past` (which consumes both `probs_past`
  # derived from `key_cache` and `value_cache`) to the cache write that
  # survives StableHLO canonicalization (`* 0` is constant-folded away).
  # Note: this must read from the internal `out_past` tensor rather than the
  # composite output `attention`, because TFLite's `ReplaceNodeWithSubgraph`
  # composite inliner only remaps `decomp_node.outputs` via `output_map` and
  # does not remap `decomp_node.inputs` that consume a subgraph output tensor.
  valid_dep = out_past.reshape(-1)[:1].abs() >= 0
  new_k = _ring_write(key_cache, key_new, start, num_valid, valid_dep, ts_idx=2)
  new_v = _ring_write(
      value_cache, value_new, start, num_valid, valid_dep, ts_idx=3
  )
  attention, new_k, new_v = builder.mark_outputs(attention, new_k, new_v)
  return attention, new_k, new_v
