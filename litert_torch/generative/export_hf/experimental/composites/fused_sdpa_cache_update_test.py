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
"""Tests for the fused SWA attention + ring buffer cache update composite."""

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.experimental.composites import fused_sdpa_cache_update as fused_lib
from litert_torch.generative.export_hf.experimental.composites import sdpa
import numpy as np
import torch
from absl.testing import absltest as googletest

_WINDOW = 8  # Ring buffer size == sliding window.
_HEADS = 2  # KV heads.
_DIM = 4


class _Case:
  """A sliding-window prefill step over a known token history."""

  def __init__(self, past, new_len, valid, g=1, seed=0):
    gen = torch.Generator().manual_seed(seed)
    total = past + new_len
    self.past, self.new_len, self.valid, self.g = past, new_len, valid, g
    # Full K/V history; positions >= past + valid are padding.
    self.k_hist = torch.randn(1, _HEADS, total, _DIM, generator=gen)
    self.v_hist = torch.randn(1, _HEADS, total, _DIM, generator=gen)
    self.query = torch.randn(1, _HEADS, g * new_len, _DIM, generator=gen)
    self.key_cache, self.value_cache = self._ring(past)
    self.key_new = self.k_hist[:, :, past:]
    self.value_new = self.v_hist[:, :, past:].transpose(-2, -1)
    self.param = torch.tensor(
        [[[[past, past + valid, past + valid, valid, 0, 0, 0]]]],
        dtype=torch.int32,
    )
    self.mask = self._mask()

  def _slot_positions(self, length):
    """Position held by each ring slot after writing tokens [0, length)."""
    pos = [-1] * _WINDOW
    for p in range(length):
      pos[p % _WINDOW] = p
    return pos

  def _ring(self, length):
    k = torch.zeros(1, _HEADS, _WINDOW, _DIM)
    v = torch.zeros(1, _HEADS, _DIM, _WINDOW)
    for s, p in enumerate(self._slot_positions(length)):
      if p >= 0:
        k[:, :, s] = self.k_hist[:, :, p]
        v[:, :, :, s] = self.v_hist[:, :, p]
    return k, v

  def _mask(self):
    """[1, 1, T, W + T] bool mask in the runtime's prefill layout."""
    t = self.new_len
    mask = torch.zeros(1, 1, t, _WINDOW + t, dtype=torch.bool)
    slot_pos = self._slot_positions(self.past)
    for i in range(t):
      q = self.past + i
      for s, p in enumerate(slot_pos):
        mask[0, 0, i, s] = 0 <= p and q - p < _WINDOW
      for j in range(t):
        mask[0, 0, i, _WINDOW + j] = j <= i and i - j < _WINDOW
    return mask

  def reference_attention(self, softcap=None):
    """Plain sliding-window attention over the linear history."""
    t = self.new_len
    out = torch.zeros_like(self.query)
    for row in range(self.g * t):
      q_pos = self.past + row % t
      keys = [p for p in range(q_pos + 1) if q_pos - p < _WINDOW]
      q = self.query[:, :, row]  # [1, H, D]
      k = self.k_hist[:, :, keys]  # [1, H, n, D]
      logits = torch.einsum("bhd,bhnd->bhn", q, k)
      if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
      probs = torch.softmax(logits, dim=-1)
      out[:, :, row] = torch.einsum("bhn,bhnd->bhd", probs, self.v_hist[:, :, keys])
    return out

  def reference_caches(self):
    return self._ring(self.past + self.valid)


class _RingLayer:

  def __init__(self, keys, values):
    self.keys, self.values = keys, values
    self.is_sliding = True
    self.max_cache_len = _WINDOW
    self.k_ts_idx, self.v_ts_idx = 2, 3


class FusedSdpaCacheUpdateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("first_prefill", 0, 4, 4, 1),
      ("partial_cache", 3, 4, 4, 1),
      ("wraps_ring", 6, 4, 4, 1),
      ("gqa_wraps_ring", 6, 4, 4, 2),
      ("longer_than_window", 5, 12, 12, 1),
      ("padding", 6, 4, 2, 1),
      ("padding_longer_than_window", 3, 12, 10, 2),
  )
  def test_matches_sliding_window_reference(self, past, new_len, valid, g):
    case = _Case(past, new_len, valid, g)
    attn, new_k, new_v = fused_lib.fused_sdpa_cache_update(
        case.query,
        case.key_cache,
        case.value_cache,
        case.key_new,
        case.value_new,
        case.mask,
        case.param,
        update_cache=True,
    )
    t = case.new_len
    # Padded query rows are don't-care; compare the valid ones.
    valid_rows = [r for r in range(g * t) if r % t < valid]
    ref = case.reference_attention()
    torch.testing.assert_close(attn[:, :, valid_rows], ref[:, :, valid_rows])
    ref_k, ref_v = case.reference_caches()
    torch.testing.assert_close(new_k, ref_k)
    torch.testing.assert_close(new_v, ref_v)

  def test_softcap(self):
    case = _Case(6, 4, 4, seed=3)
    attn, _, _ = fused_lib.fused_sdpa_cache_update(
        case.query * 4,
        case.key_cache,
        case.value_cache,
        case.key_new,
        case.value_new,
        case.mask,
        case.param,
        update_cache=False,
        softcap=2.0,
    )
    case.query = case.query * 4
    torch.testing.assert_close(attn, case.reference_attention(softcap=2.0))

  def test_readers_do_not_write(self):
    case = _Case(6, 4, 4)
    attn, new_k, new_v = fused_lib.fused_sdpa_cache_update(
        case.query,
        case.key_cache,
        case.value_cache,
        case.key_new,
        case.value_new,
        case.mask,
        case.param,
        update_cache=False,
    )
    self.assertIsNone(new_k)
    self.assertIsNone(new_v)
    torch.testing.assert_close(attn, case.reference_attention())

  @parameterized.named_parameters(
      ("partial_cache", 3, 4, 1),
      ("wraps_ring", 6, 4, 2),
      ("longer_than_window", 5, 12, 1),
  )
  def test_ring_buffer_sdpa_fused_matches_unfused(self, past, new_len, g):
    """The flag must not change results where the unfused path is exact."""
    case = _Case(past, new_len, new_len, g)
    cache_position = torch.arange(past, past + new_len, dtype=torch.int32)

    def run(use_fused):
      layer = _RingLayer(case.key_cache.clone(), case.value_cache.clone())
      encoded = sdpa.ring_buffer_sdpa(
          query=case.query,
          key_past=layer.keys,
          value_past=layer.values,
          k_ts_idx=2,
          v_ts_idx=3,
          param_tensor=case.param,
          is_global=False,
          key_states=case.key_new,
          value_states=case.value_new,
          layer=layer,
          cache_position=cache_position,
          mask=case.mask,
          use_fused_sdpa_cache_update=use_fused,
          runtime_param_tensor=case.param if use_fused else None,
      )
      return encoded, layer

    fused, fused_layer = run(True)
    unfused, unfused_layer = run(False)
    torch.testing.assert_close(fused, unfused)
    torch.testing.assert_close(fused_layer.keys, unfused_layer.keys)
    torch.testing.assert_close(fused_layer.values, unfused_layer.values)

  @parameterized.named_parameters(
      ("writer", True, 3),
      ("reader", False, 1),
  )
  def test_export_emits_composite(self, update_cache, num_outputs):
    case = _Case(6, 4, 4)

    class Wrapper(torch.nn.Module):

      def forward(self, q, kc, vc, kn, vn, mask, param):
        outs = fused_lib.fused_sdpa_cache_update(
            q, kc, vc, kn, vn, mask, param, update_cache=update_cache
        )
        return tuple(o for o in outs if o is not None)

    exported = torch.export.export(
        Wrapper(),
        (
            case.query,
            case.key_cache,
            case.value_cache,
            case.key_new,
            case.value_new,
            case.mask,
            case.param,
        ),
    )
    marks = [
        n
        for n in exported.graph.nodes
        if n.op == "call_function"
        and "mark_tensor" in str(n.target)
        and fused_lib.COMPOSITE_NAME in (str(n.args) + str(n.kwargs))
    ]
    # 7 inputs plus the outputs are marked.
    self.assertLen(marks, 7 + num_outputs)

  def test_tflite_inplace_cache_aliasing(self):
    """In-place KV cache aliasing must read old slots before overwriting them."""
    new_len, g = 4, 2

    class Wrapper(torch.nn.Module):

      def forward(self, q, kc, vc, kn, vn, mask, param):
        attn, new_k, new_v = fused_lib.fused_sdpa_cache_update(
            q, kc, vc, kn, vn, mask, param, update_cache=True
        )
        return attn, new_k, new_v

    sample = _Case(0, new_len, new_len, g=g)
    sample_args = (
        sample.query,
        sample.key_cache,
        sample.value_cache,
        sample.key_new,
        sample.value_new,
        sample.mask,
        sample.param,
    )
    edge_model = litert_torch.convert(Wrapper().eval(), sample_args)
    self.assertIn(b"odml.fused_sdpa_cache_update", edge_model.model_content())

    cm = edge_model.use_compiled_runtime()._get_compiled_model()
    sig = "serving_default"
    input_map = {
        f"args_{idx}": cm.create_input_buffer_by_name(sig, f"args_{idx}")
        for idx in range(len(sample_args))
    }
    out_attn = cm.create_output_buffer_by_name(sig, "output_0")
    # Alias output_1/output_2 to the same buffers as args_1/args_2 (key_cache
    # and value_cache), matching LiteRT-LM's in-place CPU KV cache binding.
    output_map = {
        "output_0": out_attn,
        "output_1": input_map["args_1"],
        "output_2": input_map["args_2"],
    }
    try:
      for past, valid in ((0, 4), (3, 4), (6, 4), (6, 2)):
        case = _Case(past, new_len, valid, g=g, seed=0)
        input_map["args_0"].write(np.ascontiguousarray(case.query.numpy()))
        input_map["args_1"].write(np.ascontiguousarray(case.key_cache.numpy()))
        input_map["args_2"].write(
            np.ascontiguousarray(case.value_cache.numpy())
        )
        input_map["args_3"].write(np.ascontiguousarray(case.key_new.numpy()))
        input_map["args_4"].write(np.ascontiguousarray(case.value_new.numpy()))
        input_map["args_5"].write(np.ascontiguousarray(case.mask.numpy()))
        input_map["args_6"].write(np.ascontiguousarray(case.param.numpy()))

        cm.run_by_name(sig, input_map, output_map)

        attn_np = out_attn.read(
            int(np.prod(case.query.shape)), np.float32
        ).reshape(case.query.shape)
        new_k_np = (
            input_map["args_1"]
            .read(int(np.prod(case.key_cache.shape)), np.float32)
            .reshape(case.key_cache.shape)
        )
        new_v_np = (
            input_map["args_2"]
            .read(int(np.prod(case.value_cache.shape)), np.float32)
            .reshape(case.value_cache.shape)
        )

        valid_rows = [r for r in range(g * new_len) if r % new_len < valid]
        ref_attn = case.reference_attention().numpy()
        ref_k, ref_v = case.reference_caches()
        np.testing.assert_allclose(
            attn_np[:, :, valid_rows],
            ref_attn[:, :, valid_rows],
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            new_k_np, ref_k.numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            new_v_np, ref_v.numpy(), rtol=1e-4, atol=1e-4
        )
    finally:
      for buf in input_map.values():
        buf.destroy()
      out_attn.destroy()


if __name__ == "__main__":
  googletest.main()
