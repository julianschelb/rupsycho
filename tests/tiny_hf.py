"""A tiny, fully offline Hugging Face causal language model for tests.

The model is a randomly initialised one-layer GPT-2 with a character-level tokenizer and a
chat template. It produces gibberish, but it goes through the *real* Transformers pipeline,
sampling and tokenizer code paths, so seeding and loading can be tested without any download.
"""

from __future__ import annotations

import string
from pathlib import Path

PRINTABLE = string.ascii_letters + string.digits + string.punctuation + " \n"
CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}assistant:"
)


def build_tiny_causal_lm(directory: str | Path) -> Path:
    """Create the tiny model and tokenizer and save them to ``directory``.

    Returns:
        The directory, usable as ``name_or_path`` of a local Hugging Face model config.
    """
    import torch
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    directory = Path(directory)
    specials = ["[PAD]", "[UNK]", "[EOS]"]
    vocab = {token: index for index, token in enumerate(specials + list(PRINTABLE))}

    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = CHAT_TEMPLATE

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=4096,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=vocab["[EOS]"],
        eos_token_id=vocab["[EOS]"],
        pad_token_id=vocab["[PAD]"],
    )
    model = GPT2LMHeadModel(config)

    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


TINY_PARAMETERS = {
    "max_new_tokens": 12,
    "do_sample": True,
    "temperature": 1.5,
    "top_k": 0,
    "return_full_text": False,
}
"""Generation parameters that make the tiny model sample varied text."""
