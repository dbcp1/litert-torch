# Copyright 2025 The LiteRT Torch Authors.
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
"""Optimized Cache class for HuggingFace integration.

Shape annotations used here:
  B: batch size
  K: num_key_value_heads
  G: number of KV groups
  N: number of attention heads. N // K = G
  T: target / input length
  S: sequence / context length
  H: head dimension
"""

from typing import Any, List, Tuple, Dict

import jaxtyping as jt
import litert_torch.generative.custom_ops.dynamic_update_slice as tfl_dus
from litert_torch.generative.export_hf.core import exportable_module_config
import litert_torch.generative.export_hf.core.cache_base as cache_base_lib
import torch
import torch.utils._pytree as pytree

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig


# Shape annotations for the cache entries.
KeyCache = (
    jt.Shaped[torch.Tensor, "1 BK S H"] | jt.Shaped[torch.Tensor, "1 BK H S"]
)
KeySlice = (
    jt.Shaped[torch.Tensor, "1 BK T H"] | jt.Shaped[torch.Tensor, "1 BK H T"]
)
ValueCache = (
    jt.Shaped[torch.Tensor, "1 BK H S"] | jt.Shaped[torch.Tensor, "1 BK S H"]
)
ValueSlice = (
    jt.Shaped[torch.Tensor, "1 BK H T"] | jt.Shaped[torch.Tensor, "1 BK T H"]
)


def _get_slice_indices(
    positions: jt.Int32[torch.Tensor, "1"], cache_dim: int, ts_idx: int
) -> jt.Int32[torch.Tensor, "cache_dim"]:
  """Returns the slice indices.

  Args:
    positions: The positions tensor.
    cache_dim: Rank of the cache tensor..
    ts_idx: The index of the sequence dimension in cache.

  Returns:
    The indices tensor for tfl.dynamic_update_slice.
  """

  assert ts_idx < cache_dim, "ts_idx must be less than cache_dim."
  assert ts_idx >= 0, "ts_idx must be greater than or equal to 0."

  zeros = torch.zeros((1,), dtype=torch.int32)
  indices = []
  for i in range(cache_dim):
    if i == ts_idx:
      indices.append(
          positions.reshape(
              1,
          )
      )
    else:
      indices.append(zeros)
  slice_indices = torch.cat(indices, dim=0)
  return slice_indices


def _update_kv_impl(
    key_state: KeyCache,
    value_state: ValueCache,
    k_slice: KeySlice,
    v_slice: ValueSlice,
    cache_position: jt.Int32[torch.Tensor, "T"],
    k_ts_idx: int,
    v_ts_idx: int,
):
  """Updates the cache buffer using tfl.dynamic_update_slice."""
  cache_dim = 4
  positions = cache_position[0]  # The position of the first input token.
  k_slice_indices = _get_slice_indices(positions.clone(), cache_dim, k_ts_idx)
  v_slice_indices = _get_slice_indices(positions.clone(), cache_dim, v_ts_idx)
  k = tfl_dus.dynamic_update_slice(
      key_state, k_slice, [x for x in k_slice_indices]
  )
  v = tfl_dus.dynamic_update_slice(
      value_state, v_slice, [x for x in v_slice_indices]
  )
  return k, v


class LiteRTLMCacheLayer(cache_base_lib.LiteRTLMCacheLayerMixin):
  """Optimized Cache layer class for HuggingFace integration."""

  is_compileable = True
  is_sliding = False

  def __init__(
      self,
      key_cache: KeyCache,
      value_cache: ValueCache,
      batch_size: int = 1,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      **kwargs,
  ):
    super().__init__()
    self.keys = key_cache
    self.values = value_cache
    self.k_ts_idx = k_ts_idx  # The index of the sequence dimension in K cache.
    self.v_ts_idx = v_ts_idx  # The index of the sequence dimension in V cache.
    assert k_ts_idx in [2, 3]
    assert v_ts_idx in [2, 3]
    self.is_initialized = True

    self.k_cache_shape = self.keys.shape
    self.v_cache_shape = self.values.shape
    self.max_cache_len = self.v_cache_shape[self.v_ts_idx]
    self.batch_size = batch_size
    v_head_dim_idx = 3 if self.v_ts_idx == 2 else 2
    self.head_dim = self.v_cache_shape[v_head_dim_idx]

    self.additional_states = kwargs.get("additional_states", None)

    self.cumulative_length = 0

  def get_batch_size(self) -> int:
    return self.batch_size

  def get_k_ts_idx(self) -> int:
    return self.k_ts_idx

  def get_v_ts_idx(self) -> int:
    return self.v_ts_idx

  def lazy_initialization(self, key_states: torch.Tensor):
    # Since we don't support real lazy initialization, this function could only
    # be called by Cache.early_initialization, where uses a standard cache
    # layout [batch_size, num_heads, ?, head_dim].
    # TODO(weiyiw): Implement this function.
    raise NotImplementedError(
        "Lazy initialization is not supported in LiteRTLMCacheLayer."
    )

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      *args,
      **kwargs,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    cache_kwargs = self.get_cache_runtime_args()
    seq_len = key_states.shape[2]
    self.cumulative_length += seq_len

    key_states = key_states.to(self.keys.dtype)

    value_states = value_states.to(self.values.dtype)

    if not cache_kwargs.get("kv_slice_preprocessed", False):
      if self.k_ts_idx == 3:
        key_target_shape = (1, -1, self.head_dim, seq_len)
        key_states = key_states.permute(0, 1, 3, 2).reshape(*key_target_shape)
      elif self.k_ts_idx == 2:
        key_target_shape = (1, -1, seq_len, self.head_dim)
        key_states = key_states.reshape(*key_target_shape)
      else:
        raise ValueError(f"Unsupported k_ts_idx: {self.k_ts_idx}")
      if self.v_ts_idx == 3:
        value_target_shape = (1, -1, self.head_dim, seq_len)
        value_states = value_states.permute(0, 1, 3, 2).reshape(
            *value_target_shape
        )
      elif self.v_ts_idx == 2:
        value_target_shape = (1, -1, seq_len, self.head_dim)
        value_states = value_states.reshape(*value_target_shape)
      else:
        raise ValueError(f"Unsupported v_ts_idx: {self.v_ts_idx}")

    cache_position: jt.Int32[torch.Tensor, "T"] = cache_kwargs.get(
        "cache_position"
    )
    assert (
        cache_position is not None
    ), "For export, cache position should always be set."
    self.keys, self.values = _update_kv_impl(
        self.keys,
        self.values,
        key_states,
        value_states,
        cache_position,
        self.k_ts_idx,
        self.v_ts_idx,
    )
    return self.keys, self.values

  def get_mask_sizes(self, cache_position: torch.Tensor):
    """Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for."""
    kv_offset = 0
    kv_length = self.max_cache_len
    return kv_length, kv_offset

  def get_seq_length(self) -> int:
    return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

  def get_max_cache_shape(self) -> int:
    return self.max_cache_len

  @classmethod
  def _infer_cache_shape_from_config(
      cls,
      model_config,
      layer_index,
      export_config: ExportableModuleConfig,
  ):
    """Infers the KV cache shape from the model config."""
    del layer_index  # Unused.
    cache_length = export_config.cache_length
    batch_size = export_config.batch_size
    k_ts_idx = export_config.k_ts_idx
    v_ts_idx = export_config.v_ts_idx
    num_kv_heads = model_config.num_key_value_heads
    embed_size_per_head = (
        getattr(model_config, "head_dim", None)
        or model_config.hidden_size // model_config.num_attention_heads
    )

    if k_ts_idx == 2:
      k_cache_shape = (
          1,
          batch_size * num_kv_heads,
          cache_length,
          embed_size_per_head,
      )
    elif k_ts_idx == 3:
      k_cache_shape = (
          1,
          batch_size * num_kv_heads,
          embed_size_per_head,
          cache_length,
      )
    else:
      raise ValueError(f"Unsupported k_ts_idx: {k_ts_idx}")
    if v_ts_idx == 2:
      v_cache_shape = (
          1,
          batch_size * num_kv_heads,
          cache_length,
          embed_size_per_head,
      )
    elif v_ts_idx == 3:
      v_cache_shape = (
          1,
          batch_size * num_kv_heads,
          embed_size_per_head,
          cache_length,
      )
    else:
      raise ValueError(f"Unsupported v_ts_idx: {v_ts_idx}")
    return k_cache_shape, v_cache_shape

  @classmethod
  def create_from_config(
      cls,
      model_config,
      layer_index,
      export_config: ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLMCacheLayer":
    """Creates a KV cache from the model config."""
    k_cache_shape, v_cache_shape = cls._infer_cache_shape_from_config(
        model_config, layer_index, export_config
    )
    keys = torch.zeros(k_cache_shape, dtype=torch.float32)
    values = torch.zeros(v_cache_shape, dtype=torch.float32)
    return cls(
        keys,
        values,
        k_ts_idx=export_config.k_ts_idx,
        v_ts_idx=export_config.v_ts_idx,
        **kwargs,
    )


@cache_base_lib.register_cache_implementation
class LiteRTLMCache(cache_base_lib.LiteRTLMCacheMixin):
  """Optimized Cache class for HuggingFace integration."""

  @classmethod
  def create_from_config(
      cls,
      model_config,
      export_config: ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLMCache":
    """Creates a KV cache from the model config."""
    num_layers = model_config.num_hidden_layers
    layers = []
    for layer_index in range(num_layers):
      layers.append(
          LiteRTLMCacheLayer.create_from_config(
              model_config,
              layer_index,
              export_config,
              **kwargs,
          )
      )
    return cls(layers)


def _flatten_kvc_t(
    kvc: LiteRTLMCache,
) -> Tuple[List[torch.Tensor], Tuple[List[str], Tuple[int, int, int, int]]]:
  """Flattens the cache into a list of tensors."""
  flattened = []
  flat_names = []
  num_layers = len(kvc.layers)
  layer_0 = kvc.layers[0]
  assert isinstance(layer_0, cache_base_lib.LiteRTLMCacheLayerMixin)
  batch_size = layer_0.get_batch_size()
  k_ts_idx = layer_0.get_k_ts_idx()
  v_ts_idx = layer_0.get_v_ts_idx()
  for i, layer in enumerate(kvc.layers):
    flattened.append(layer.keys)
    flat_names.append(f"k_{i}")
    flattened.append(layer.values)
    flat_names.append(f"v_{i}")
  return flattened, (flat_names, (batch_size, num_layers, k_ts_idx, v_ts_idx))


def _unflatten_kvc_t(
    values: List[torch.Tensor],
    context: Tuple[List[str], Tuple[int, int, int, int]],
) -> LiteRTLMCache:
  """Unflattens the cache from a list of tensors."""
  flat_names = context[0]
  batch_size, num_layers, k_ts_idx, v_ts_idx = context[1]
  layers = []
  for i in range(num_layers):
    k_cache_idx = flat_names.index(f"k_{i}")
    v_cache_idx = flat_names.index(f"v_{i}")
    layers.append(
        LiteRTLMCacheLayer(
            key_cache=values[k_cache_idx],
            value_cache=values[v_cache_idx],
            batch_size=batch_size,
            k_ts_idx=k_ts_idx,
            v_ts_idx=v_ts_idx,
        )
    )
  obj = LiteRTLMCache(layers)
  return obj


def _flatten_kvc_t_with_keys(
    kvc: LiteRTLMCache,
):
  flattened, (flat_names, _) = _flatten_kvc_t(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


pytree.register_pytree_node(
    LiteRTLMCache,
    _flatten_kvc_t,
    _unflatten_kvc_t,
    flatten_with_keys_fn=_flatten_kvc_t_with_keys,
    serialized_type_name="",
)
