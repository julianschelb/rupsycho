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
must be pulled on the Ollama server first (`ollama pull llama3`). The Google entry shows an
`api_key` placeholder because that is currently where the key has to be given (see below);
keep such a file private.

### API keys

Keep keys out of configuration files that you commit or share.

| Back-end             | Where the key comes from                                                              |
| -------------------- | ------------------------------------------------------------------------------------- |
| `openai`             | The `OPENAI_API_KEY` environment variable, or `api_key` in the configuration          |
| `google`             | `api_key` in the configuration. The `GOOGLE_API_KEY` variable is only used by models you create in Python and add with `add_model` |
| `deepseek`           | The `DEEPSEEK_API_KEY` environment variable                                            |
| `remote_huggingface` | The `HUGGINGFACEHUB_API_TOKEN` (or `HF_TOKEN`) environment variable                    |

To avoid keys in configurations altogether, create the model in Python, where the standard
environment variables apply, and add it to the experiment:

```python
from langchain_openai import ChatOpenAI

experiment.add_model(ChatOpenAI(model="gpt-4o-mini", temperature=0.7), identifier="gpt")
```

LangChain replaces the API key of a model added this way by a placeholder when the model
is stored in the experiment, so `export_to_file` does not write it. Keys written in the
`models` section of a configuration are written to the exported file as they are.

## Seeds

Each entry of `parameters.seeds` is passed to the model call as `seed=<int>` and gives one
run of the experiment. What the seed does is up to the back-end:

| Back-end                    | Effect of the seed                                                                |
| --------------------------- | --------------------------------------------------------------------------------- |
| `openai` (and compatible)   | Forwarded to the API, which samples deterministically on a best-effort basis      |
| `google`                    | Google models do not accept a seed, so none is passed and the seed list only repeats the run |
| `local_huggingface`         | The argument is currently ignored by the pipeline; set `"do_sample": false` for identical reruns |
| others                      | Forwarded as `seed=<int>`; the effect depends on the model and its client library |

## Several models in one experiment

All listed models are run one after another on every item, persona and seed. This makes
cross-model comparisons a matter of adding entries to the `models` mapping. Each model has
its own `parameters`, so settings such as the temperature can differ between entries; give
every entry a distinct identifier, because it labels the results.
