# ===========================================================================
#               Data Model: Answer Generation Parameters
# ===========================================================================
# This file contains the data model for the parameters used in answer generation.


from random import randint
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class Parameters(BaseModel):
    """Represents the parameters for a model. """

    # ------------------- Experiment -------------------

    seeds: Optional[List[str]] = Field(
        default=[str(randint(0, 999999))],
        description="A list of seeds for random number generation."
    )

    # ------------------- Huggingface -------------------

    min_new_tokens: Optional[int] = Field(
        default=None,
        description="The minimum number of new tokens to generate."
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of new tokens to generate."
    )
    temperature: Optional[float] = Field(
        default=None,
        description="The temperature for sampling."
    )
    do_sample: Optional[bool] = Field(
        default=None,
        description="Whether to sample or not during generation."
    )
    top_k: Optional[int] = Field(
        default=None,
        description="The number of highest probability vocabulary tokens to keep for top-k filtering."
    )
    top_p: Optional[float] = Field(
        default=None,
        description="The cumulative probability of parameter choices to keep for nucleus sampling."
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        description="Penalizes repeated tokens."
    )
    length_penalty: Optional[float] = Field(
        default=None,
        description="Adjusts the output length."
    )
    no_repeat_ngram_size: Optional[int] = Field(
        default=None,
        description="Ensures no n-grams of this size are repeated in the output."
    )
    early_stopping: Optional[bool] = Field(
        default=None,
        description="Stops generation as soon as a specified condition is met."
    )
    num_beams: Optional[int] = Field(
        default=None,
        description="The number of beams to use in beam search."
    )
    num_return_sequences: Optional[int] = Field(
        default=None,
        description="The number of independently computed returned sequences for each prompt."
    )
    bad_words_ids: Optional[List[List[int]]] = Field(
        default=None,
        description="List of token IDs that should not appear in the generated text."
    )
    pad_token_id: Optional[int] = Field(
        default=None,
        description="The ID of the token to use for padding."
    )
    bos_token_id: Optional[int] = Field(
        default=None,
        description="The ID of the token to use as the beginning of the sequence token."
    )
    eos_token_id: Optional[int] = Field(
        default=None,
        description="The ID of the token to use as the end of the sequence token."
    )
    decoder_start_token_id: Optional[int] = Field(
        default=None,
        description="The ID of the token to use to start decoding."
    )
    forced_bos_token_id: Optional[int] = Field(
        default=None,
        description="The ID of the token to force as the first generated token."
    )
    forced_eos_token_id: Optional[int] = Field(
        default=None,
        description="The ID of the token to force as the last generated token."
    )
    diversity_penalty: Optional[float] = Field(
        default=None,
        description="Encourages more diverse outputs when using beam search."
    )
    prefix_allowed_tokens_fn: Optional[Any] = Field(  # Callable type can't be used directly with Pydantic
        default=None,
        description="A callable that returns a list of allowed tokens for the next step."
    )
    output_attentions: Optional[bool] = Field(
        default=None,
        description="Whether to return attention scores."
    )
    output_hidden_states: Optional[bool] = Field(
        default=None,
        description="Whether to return hidden states."
    )
    output_scores: Optional[bool] = Field(
        default=None,
        description="Whether to return prediction scores."
    )

    # ------------------- OpenAI -------------------

    logprobs: Optional[int] = Field(
        None,
        description="Include the log probabilities on the most likely tokens."
    )
    stop: Optional[List[str]] = Field(
        None,
        description="A sequence or list of sequences where the API should stop generating further tokens."
    )
    presence_penalty: Optional[float] = Field(
        None,
        description="Penalizes new tokens based on their presence in the text so far."
    )
    frequency_penalty: Optional[float] = Field(
        None,
        description="Penalizes new tokens based on their frequency in the text so far."
    )
    best_of: Optional[int] = Field(
        None,
        description="Generates best_of completions and returns the best one."
    )
    logit_bias: Optional[Dict[int, float]] = Field(
        None,
        description="Adjust the likelihood of specific tokens appearing in the output."
    )
    user: Optional[str] = Field(
        None,
        description="A unique identifier representing your end-user."
    )

    # ------------------- Ollama -------------------

    max_context_length: Optional[int] = Field(
        None,
        description="Maximum length of the context (input + output) in tokens."
    )
    system_message: Optional[str] = Field(
        None,
        description="An optional message that sets the behavior or instructions for the model."
    )
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="A dictionary of metadata to attach to the request."
    )
    tags: Optional[List[str]] = Field(
        None,
        description="A list of tags or labels associated with the request."
    )
    user_id: Optional[str] = Field(
        None,
        description="A unique identifier for the user making the request."
    )

    class Config:
        extra = 'allow'  # Allow extra fields
