# Security Policy

If you discover a security vulnerability, please **do not open a public issue**. Report it
privately through GitHub's
[security advisories](https://github.com/julianschelb/rupsycho/security/advisories/new)
or by email to <julian.schelb@uni-konstanz.de>.

Note that experiment configurations may contain API keys, and `export_to_file` writes the model
configurations, including any `api_key` set there, into the exported file. Never commit or share
such files; prefer environment variables, see the
[Models guide](https://julianschelb.github.io/rupsycho/tutorials/models/#api-keys).
