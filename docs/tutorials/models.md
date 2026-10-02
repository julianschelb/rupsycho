# Models

Models are declared in the `models` section, or added in Python with
`experiment.add_model(model, identifier=...)` (any LangChain chat model or LLM works).
The field reference is in the [Configuration Reference](../configuration.md#models).

## Local Hugging Face

```json
"models": {
  "smollm": {
    "type": "local_huggingface",
    "name_or_path": "HuggingFaceTB/SmolLM-1.7b-Instruct",
    "device_map": "cpu",
    "task": "text-generation",
    "parameters": {
      "max_new_tokens": 64,
      "temperature": 1.0,
      "do_sample": true,
      "return_full_text": false
    }
  }
}
```

The model is downloaded when the experiment first needs it. `parameters` are arguments of the
Transformers `pipeline`. Local models are loaded as causal language models and used through
a chat interface, so the tokenizer needs a chat template (instruction-tuned models have one).
For gated models, log in with `hf auth login` or set `HF_TOKEN`.

For large models install the `quantization` extra and pass a `bitsandbytes_config`
(forwarded to `transformers.BitsAndBytesConfig`).

## Hosted APIs

```json
"models": {
  "gpt": {"type": "openai", "name_or_path": "gpt-4o-mini", "parameters": {"temperature": 0.7}},
  "gemini": {"type": "google", "name_or_path": "gemini-flash-latest", "api_key": "<google-api-key>"},
  "deepseek": {"type": "deepseek", "name_or_path": "deepseek-chat"},
  "zephyr": {"type": "remote_huggingface", "repo_id": "HuggingFaceH4/zephyr-7b-beta", "task": "text-generation"},
  "llama": {"type": "ollama", "model": "llama3", "base_url": "http://localhost:11434"}
}
```

The model identifiers are examples; use any identifier your provider offers. Ollama models
must be pulled on the Ollama server first (`ollama pull llama3`). Each back-end needs its extra
(`huggingface`, `openai`, `ollama`, `google`, `deepseek`, or `models` for all); using a back-end
without it raises an `ImportError` that names the extra.

### API keys

Keep keys out of configuration files that you commit or share. When a configuration contains no
key, the provider's environment variable is used:

| Back-end             | Environment variable                          |
| -------------------- | --------------------------------------------- |
| `openai`             | `OPENAI_API_KEY`                              |
| `google`             | `GOOGLE_API_KEY`                              |
| `deepseek`           | `DEEPSEEK_API_KEY`                            |
| `remote_huggingface` | `HUGGINGFACEHUB_API_TOKEN` (or `HF_TOKEN`)    |

`api_key` (and `huggingfacehub_api_token`) in a configuration are accepted but treated as
secrets: they never show up in `repr` or in exports (`to_config`, `export_to_file` write
`**********`, which is read back as "no key", so the environment variable applies again).

You can also create the model in Python and add it to the experiment:

```python
from langchain_openai import ChatOpenAI

experiment.add_model(ChatOpenAI(model="gpt-4o-mini", temperature=0.7), identifier="gpt")
```

## Pinning versions

For Hugging Face models, set `revision` (a commit hash) so that the exact weights are part of
your configuration; `cache_dir` and `huggingfacehub_api_token` (gated models) are honoured too.

## Seeds

Each entry of `parameters.seeds` gives one repetition of the experiment and is applied in the way
the back-end supports: `transformers.set_seed` before every call for local Hugging Face models,
the `seed` parameter for OpenAI/DeepSeek and Hugging Face endpoints, the `seed` option for
Ollama. Google Gemini cannot be seeded (a warning is emitted once). The full table, the
caveats and how to verify reproducibility are on the [Reproducibility](reproducibility.md) page.

## Several models in one experiment

All listed models are run one after another on every item, persona and seed. This makes
cross-model comparisons a matter of adding entries to the `models` mapping. Each model has
its own `parameters`, so settings such as the temperature can differ between entries; give
every entry a distinct identifier, because it labels the results.
