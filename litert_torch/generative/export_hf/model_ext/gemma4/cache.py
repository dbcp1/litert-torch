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
"""Cache module for Gemma4."""

import copy
from typing import List, Tuple
from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import cache_base as cache_base_lib
from litert_torch.generative.export_hf.core import exportable_module_config
import torch
import torch.utils._pytree as pytree


class LiteRTLMCacheLayerForGemma4(cache_lib.LiteRTLMCacheLayer):

  @classmethod
  def _infer_cache_shape_from_config(
      cls,
      model_config,
      layer_index,
      export_config: exportable_module_config.ExportableModuleConfig,
  ):
    """Infers the KV cache shape from the model config."""
    cache_length = export_config.cache_length
    batch_size = export_config.batch_size
    k_ts_idx = export_config.k_ts_idx
    v_ts_idx = export_config.v_ts_idx
    num_kv_heads = model_config.num_key_value_heads
    if hasattr(model_config, "num_global_key_value_heads"):
      layer_type = model_config.layer_types[layer_index]
      if layer_type == "full_attention":
        num_kv_heads = model_config.num_global_key_value_heads or num_kv_heads
    embed_size_per_head = (
        getattr(model_config, "head_dim", None)
        or model_config.hidden_size // model_config.num_attention_heads
    )
    if hasattr(model_config, "global_head_dim"):
      layer_type = model_config.layer_types[layer_index]
      if layer_type == "full_attention":
        embed_size_per_head = (
            model_config.global_head_dim or embed_size_per_head
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


@cache_base_lib.register_cache_implementation
class LiteRTCacheForGemma4(cache_lib.LiteRTLMCache):

  @classmethod
  def create_from_config(
      cls,
      model_config,
      export_config: exportable_module_config.ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTCacheForGemma4":
    """Creates a KV cache from the model config."""
    num_layers = model_config.num_hidden_layers
    num_shared_layers = model_config.num_kv_shared_layers
    layers = []
    for layer_index in range(num_layers - num_shared_layers):
      layers.append(
          LiteRTLMCacheLayerForGemma4.create_from_config(
              model_config,
              layer_index,
              export_config,
              **kwargs,
          )
      )
    return cls(layers)

  def insert_dummy_cache_layers(self, model_config):
    num_layers = model_config.num_hidden_layers
    num_shared_layers = model_config.num_kv_shared_layers
    num_unshared_layers = num_layers - num_shared_layers
    assert len(self.layers) == num_unshared_layers
    for i in range(num_shared_layers):
      self.layers.append(copy.copy(self.layers[i % num_unshared_layers]))
    return self

  def remove_dummy_cache_layers(self, model_config):
    num_layers = model_config.num_hidden_layers
    num_shared_layers = model_config.num_kv_shared_layers
    num_unshared_layers = num_layers - num_shared_layers
    assert len(self.layers) == num_layers
    self.layers = self.layers[:num_unshared_layers]
    return self


def _flatten_kvc_t(
    kvc: LiteRTCacheForGemma4,
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
) -> LiteRTCacheForGemma4:
  """Unflattens the cache from a list of tensors."""
  flat_names = context[0]
  batch_size, num_layers, k_ts_idx, v_ts_idx = context[1]
  layers = []
  for i in range(num_layers):
    k_cache_idx = flat_names.index(f"k_{i}")
    v_cache_idx = flat_names.index(f"v_{i}")
    layers.append(
        LiteRTLMCacheLayerForGemma4(
            key_cache=values[k_cache_idx],
            value_cache=values[v_cache_idx],
            batch_size=batch_size,
            k_ts_idx=k_ts_idx,
            v_ts_idx=v_ts_idx,
        )
    )
  obj = LiteRTCacheForGemma4(layers)
  return obj


def _flatten_kvc_t_with_keys(
    kvc: LiteRTCacheForGemma4,
):
  flattened, (flat_names, _) = _flatten_kvc_t(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


pytree.register_pytree_node(
    LiteRTCacheForGemma4,
    _flatten_kvc_t,
    _unflatten_kvc_t,
    flatten_with_keys_fn=_flatten_kvc_t_with_keys,
    serialized_type_name="",
)
