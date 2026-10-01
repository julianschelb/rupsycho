# Models

Models are declared in the `models` section, or added in Python with
`experiment.add_model(model, identifier=...)` (any LangChain runnable works).

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

For large models install the `quantization` extra and pass a `bitsandbytes_config`
(forwarded to `transformers.BitsAndBytesConfig`).

## Hosted APIs

```json
"models": {
  "gpt": {"type": "openai", "name_or_path": "gpt-4o-mini", "parameters": {"temperature": 0.7}},
  "gemini": {"type": "google", "name_or_path": "gemini-1.5-flash"},
  "deepseek": {"type": "deepseek", "name_or_path": "deepseek-chat"},
  "llama": {"type": "ollama", "model": "llama3", "base_url": "http://localhost:11434"}
}
```

Prefer environment variables (`OPENAI_API_KEY`, …) over `api_key` fields so keys never end
up in a shared configuration.

!!! note
    Google models do not accept a seed, so the seed list has no effect on them.

## Several models in one experiment

All listed models are run one after another on every item, persona and seed. This makes
cross-model comparisons a matter of adding entries to the `models` mapping.
