# Security Policy

## Supported versions

Only the latest minor release receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |
| < 0.1   | no        |

## Reporting a vulnerability

R.U.Psycho loads experiment configurations, runs language models (locally or through
provider APIs) and writes the answers to files. If you find a security-relevant issue, for
example a crafted configuration that executes code you would not expect from a data file,
leaked API keys, or a vulnerable dependency, please report it **privately** and do not
open a public issue:

- Use GitHub's private vulnerability reporting on this repository
  ([Report a vulnerability](https://github.com/julianschelb/rupsycho/security/advisories/new),
  Security tab, Advisories), or
- send an e-mail to <julian.schelb@uni-konstanz.de>.

Please include the `rupsycho` and Python versions, the affected component and, if possible,
a minimal configuration that reproduces the problem (without real API keys). You can expect
an acknowledgement within a week and a fix or a mitigation plan within 30 days. Fixes are
released as a new patch version of the latest minor release.

## API keys in experiment configurations

Experiment configurations are meant to be shared with papers and repositories, but some
model configurations have fields for secrets: `api_key` (OpenAI, Google, DeepSeek) and
`huggingfacehub_api_token` (local Hugging Face models).

- **Never commit keys.** Leave the fields unset: the provider libraries then read the key
  from the environment (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`,
  `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`).
- Keys that you do set are held as secrets: they are not shown in `repr`, and
  `ExperimentDocument.to_config` and `export_to_file` write them masked as `**********`,
  which is read back as "no key" (the environment variable is used). An exported configuration
  therefore contains no keys; a key that you typed into a JSON file by hand stays in that file.
- The repository's pre-commit hook `no-api-keys-in-configs` rejects JSON files with a real
  `api_key` or `huggingfacehub_api_token` value (empty values, `<placeholders>` and the
  `**********` mask are fine), and `detect-private-key` rejects files that contain private
  keys. Install the hooks with `pre-commit install`.
- If a key was committed, pasted into an issue or shown in a log, **revoke it at the
  provider right away**. Deleting it from the Git history is not enough.
- Do not paste unredacted configurations, tracebacks or logs into issues.

## Treat configurations like code

Loading a configuration builds models and may download and run third-party code and
weights, so only load configurations (and model repositories) you trust:

- Models of type `langchain` are deserialized by LangChain, which can instantiate provider
  classes and read secrets from the environment.
- Local and remote Hugging Face models, and the model-based validators and judges, download
  weights from the Hugging Face Hub. Pin `revision` to a commit hash and review the
  repository before you enable options such as `trust_remote_code`.
- The configurator app (`rup-configurator`) sends the text of an imported questionnaire PDF
  to the OpenAI API when you use the LLM-assisted import. Do not upload confidential
  documents.
- Callbacks write the answers of the models to files you choose. Treat those files as
  untrusted text when you process them further.
