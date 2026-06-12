---
name: litert-model-equivalence-test
description: >-
  Validates equivalence between LiteRT models (litert_lm) and PyTorch models
  (transformers). Use when you need to verify that an exported LiteRT model
  produces the same outputs as the original Hugging Face model. Supports
  multi-turn conversations and custom prompts.
---

# LiteRT Model Equivalence Test

This skill provides instructions for running equivalence tests between LiteRT
models and their PyTorch source models.

## Usage

Use the `equivalence_test` script to compare the outputs of a Hugging Face model
and its exported LiteRT version.

### Running the Test

Run the test using `bazel run` from your workspace:

```bash
bazel run \
  //third_party/py/litert_torch/generative/export_hf/experimental/validation:equivalence_test \
  -- \
  --model_id={model_id} \
  [--prompt={prompt}] \
  [--prompt_file={prompt_file}] \
  [--max_new_tokens={max_new_tokens}] \
  [--max_num_tokens={max_num_tokens}] \
  [--work_dir={work_dir}] \
  [--externalize_embedder] \
  [--single_token_embedder] \
  [--split_cache] \
  [--backend={backend}]
```

### Flags

*   `--model_id`: The Hugging Face model ID to validate (e.g.,
    `google/gemma-3-270m-it`).
*   `--prompt`: Prompt to test. Specify multiple times for multi-turn
    conversations.
*   `--prompt_file`: Path to a file containing one (complex) prompt. Overrides
    `--prompt`.
*   `--max_new_tokens`: Maximum new tokens to generate per turn (default: 20).
*   `--max_num_tokens`: KV cache length for the model (default: 2048).
*   `--work_dir`: Base directory for model export. If not specified, a
    temporary directory under `HOME` is used.
*   `--externalize_embedder`: Externalize the embedder during export (default:
    False).
*   `--single_token_embedder`: Use single token embedder during export (default:
    False).
*   `--split_cache`: Split KV cache during export (default: False).
*   `--backend`: Hardware backend to use for LiteRT LM (cpu | npu, default:
    cpu).

### Examples

#### Single-turn test with custom prompt:
```bash
bazel run \
  //third_party/py/litert_torch/generative/export_hf/experimental/validation:equivalence_test \
  -- \
  --model_id=google/gemma-3-270m-it \
  --prompt="What is the capital of France?"
```

#### Multi-turn test:
```bash
bazel run \
  //third_party/py/litert_torch/generative/export_hf/experimental/validation:equivalence_test \
  -- \
  --model_id=google/gemma-3-270m-it \
  --prompt="What's the capital of France?" \
  --prompt="How about Germany?"
```

#### Testing with a prompt file:
```bash
bazel run \
  //third_party/py/litert_torch/generative/export_hf/experimental/validation:equivalence_test \
  -- \
  --model_id=google/gemma-3-270m-it \
  --prompt_file=/path/to/prompts.txt
```

#### Testing with externalized embedder:
```bash
bazel run \
  //third_party/py/litert_torch/generative/export_hf/experimental/validation:equivalence_test \
  -- \
  --model_id=google/gemma-3-270m-it \
  --externalize_embedder \
  --single_token_embedder
```

#### Testing NPU export variant:
```bash
bazel run \
  //third_party/py/litert_torch/generative/export_hf/experimental/validation:equivalence_test \
  -- \
  --model_id=google/gemma-3-270m-it \
  --externalize_embedder \
  --split_cache \
  --backend=npu
```
