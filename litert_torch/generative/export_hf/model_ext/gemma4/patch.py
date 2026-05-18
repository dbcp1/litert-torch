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
"""Patches for Gemma4 model for on-device deployment."""

import contextlib
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.layers import normalization
import torch
import transformers


class Gemma4RMSNorm(torch.nn.Module):
  """RMSNorm Layer."""

  def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
    """RMSNorm Layer."""
    super().__init__()
    self.with_scale = with_scale

    if self.with_scale:
      self.weight = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
    else:
      self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    self.variance_epsilon = eps
    self.hidden_size = dim

  def forward(self, hidden_states):
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight
        if self.with_scale
        else torch.ones((self.hidden_size,), dtype=torch.float32),
        self.variance_epsilon,
        torch.ones((self.hidden_size,), dtype=torch.float32),
    )

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


try:
  from transformers.models.gemma4 import modeling_gemma4  # pylint: disable=g-import-not-at-top

  Gemma4VisionEncoder = modeling_gemma4.Gemma4VisionEncoder
  Gemma4VisionPatchEmbedder = modeling_gemma4.Gemma4VisionPatchEmbedder
  Gemma4VisionPooler = modeling_gemma4.Gemma4VisionPooler
except ImportError:
  Gemma4VisionEncoder = torch.nn.Module
  Gemma4VisionPatchEmbedder = torch.nn.Module
  Gemma4VisionPooler = torch.nn.Module


class LiteRTGemma4VisionPatchEmbedder(Gemma4VisionPatchEmbedder):
  """LiteRT Gemma4 Vision Patch Embedder."""

  def _position_embeddings(
      self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
  ) -> torch.Tensor:
    clamped_positions = pixel_position_ids.clamp(min=0)

    classes = torch.arange(
        self.position_embedding_size,
        device=clamped_positions.device,
        dtype=torch.int32,
    )
    clamped_positions_x = clamped_positions[..., 0]
    clamped_positions_y = clamped_positions[..., 1]
    one_hot_x = clamped_positions_x.unsqueeze(-1) == classes
    one_hot_y = clamped_positions_y.unsqueeze(-1) == classes
    one_hot_x = one_hot_x.to(self.position_embedding_table.dtype)
    one_hot_y = one_hot_y.to(self.position_embedding_table.dtype)

    # AI Edge Quantizer crashes on BMM weight quant.
    table_x = self.position_embedding_table[0]
    table_y = self.position_embedding_table[1]
    x_embeddings = one_hot_x @ table_x
    y_embeddings = one_hot_y @ table_y
    position_embeddings = x_embeddings + y_embeddings
    position_embeddings = torch.where(
        padding_positions.unsqueeze(-1), 0.0, position_embeddings
    )
    return position_embeddings


class LiteRTGemma4VisionPooler(Gemma4VisionPooler):
  """LiteRT Gemma4 Vision Pooler."""

  def _avg_pool_by_positions(
      self,
      hidden_states: torch.Tensor,
      pixel_position_ids: torch.Tensor,
      length: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """2D spatial pooling according to patch positions.

    Pools the input tokens by averaging patches within a `k^2` grid, where `k`
    is determined by the ratio between input and output lengths

    Args:
      hidden_states: The input hidden states.
      pixel_position_ids: The pixel position ids.
      length: The output length.

    Returns:
      The pooled hidden states and the mask.
    """
    input_seq_len = hidden_states.shape[1]
    k = int((input_seq_len // length) ** 0.5)
    k_squared = k**2
    if k_squared * length != input_seq_len:
      raise ValueError(
          f"Cannot pool {hidden_states.shape} to {length}: {k=}^2 times"
          f" {length=} must be {input_seq_len}."
      )

    clamped_positions = pixel_position_ids.clamp(min=0)
    max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
    kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]

    classes = torch.arange(length, device=kernel_idxs.device, dtype=torch.int32)
    weights = (kernel_idxs.int().unsqueeze(-1) == classes).float() / k_squared

    output = weights.transpose(1, 2) @ hidden_states.float()
    mask = torch.logical_not((weights == 0).all(dim=1))
    return output.to(hidden_states.dtype), mask


class LiteRTGemma4VisionEncoder(Gemma4VisionEncoder):
  """LiteRT Gemma4 Vision Encoder."""

  def forward(
      self,
      inputs_embeds: torch.Tensor,
      attention_mask: torch.Tensor,
      pixel_position_ids: torch.LongTensor | None = None,
      **kwargs,
  ) -> transformers.modeling_outputs.BaseModelOutputWithPast:
    num_seq = attention_mask.shape[1]
    attention_mask = torch.zeros((1, 1, num_seq, num_seq), dtype=torch.float32)

    # embed positions
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)

    # decoder layers
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
      hidden_states = decoder_layer(
          hidden_states,
          attention_mask=attention_mask,
          position_embeddings=position_embeddings,
          position_ids=pixel_position_ids,
          **kwargs,
      )

    return transformers.modeling_outputs.BaseModelOutputWithPast(
        last_hidden_state=hidden_states
    )


# pytype: disable=import-error
@patches_lib.register_patch(["gemma4"])
@contextlib.contextmanager
def gemma4_litert_patch():
  """Gemma4 patch."""
  print("Gemma4 patch applied.")
  from transformers.models.gemma4 import modeling_gemma4  # pylint: disable=g-import-not-at-top

  original_norm = modeling_gemma4.Gemma4RMSNorm
  modeling_gemma4.Gemma4RMSNorm = Gemma4RMSNorm

  original_vision_encoder = modeling_gemma4.Gemma4VisionEncoder
  modeling_gemma4.Gemma4VisionEncoder = LiteRTGemma4VisionEncoder

  original_patch_embedder = modeling_gemma4.Gemma4VisionPatchEmbedder
  modeling_gemma4.Gemma4VisionPatchEmbedder = LiteRTGemma4VisionPatchEmbedder

  original_pooler = modeling_gemma4.Gemma4VisionPooler
  modeling_gemma4.Gemma4VisionPooler = LiteRTGemma4VisionPooler

  try:
    yield
  finally:
    modeling_gemma4.Gemma4RMSNorm = original_norm
    modeling_gemma4.Gemma4VisionEncoder = original_vision_encoder
    modeling_gemma4.Gemma4VisionPatchEmbedder = original_patch_embedder
    modeling_gemma4.Gemma4VisionPooler = original_pooler
# pytype: enable=import-error
