import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Implementing llm watermarking

    Plan:
    - Define the interface of a llm: receive prompt -> returns logits
    - Define the interface of a watermarker: receive logits -> returns logits
    - Define the interface of a sampler: receive logits -> returns a token

    Difficulties:
    - How do I get the logits from a llm? use transformers library from hf but how?
    - How do i correctly implement the watermark? Should i watermark special tokens like spaces and points?

    First step: define the interfaces
    """)
    return


@app.cell
def _():
    import abc
    import random

    import numpy as np

    from torch import FloatTensor, Tensor


    class TextGenerator(abc.ABC):
        """Abstract class representing an LLM generating text by returning logits"""

        @abc.abstractmethod
        def next_token_logits(self, prompt: str) -> list[float]:
            """
            Generates next token's logits

            Args:
                prompt (str): Prompt given to the model.

            Returns:
                list[float]: Logits.
            """
            pass


    class SoftWaterMarker:
        def __init__(
            self, seed: int | None, vocabulary_size: int, ratio: float, delta: float
        ) -> None:
            self.seed = seed
            self.delta = delta
            generator = random.Random(seed)
            shuffled_indices = [i for i in range(vocabulary_size)]
            generator.shuffle(shuffled_indices)
            green_list_size = int(vocabulary_size * ratio)
            # For now we create the green list in advanced.
            # TODO: use the hash of the token t_0 to seed the generator at each token.
            self.green_list = shuffled_indices[:green_list_size]
            self.red_list = shuffled_indices[green_list_size:]

        def update_logits(self, logits: Tensor) -> Tensor:
            """
            Updates the logits of the green list by adding delta to their value.

            Args:
                logits (list[float]): Logits predicted by the model.

            Returns:
                tuple[list[float], list[int], list[int]]: Updated logits, green list and red list.
            """
            new_logits: Tensor = logits.clone()
            for i in self.green_list:
                new_logits[:,-1j,i] += self.delta
            return new_logits


    return FloatTensor, SoftWaterMarker


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using a real model
    """)
    return


@app.cell
def _(FloatTensor):
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load LFM2.5 (choose your variant)
    model_name = "LiquidAI/LFM2.5-350M"  # or other variants
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype="bfloat16"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer
    text = "Hello, how are you?"
    inputs: dict[str, torch.Tensor] = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs: CausalLMOutputWithPast = model(**inputs)

    logits: FloatTensor = outputs.logits  # ty:ignore[invalid-assignment]
    print(logits.shape)  # [batch_size, sequence_length, vocab_size]
    return logits, torch


@app.cell
def _(SoftWaterMarker, logits: "FloatTensor", torch):
    vocabulary_size = logits.shape[2]
    watermaker = SoftWaterMarker(seed=42, vocabulary_size=vocabulary_size,ratio=0.5, delta=2)
    updated_logits = watermaker.update_logits(logits) # here we could optimize by taking only the last logits
    updated_logits = torch.softmax(updated_logits,dim=2)
    assert updated_logits[:,-1,:].sum() == 1.0, f"sum of logits should be 1 but got {updated_logits[:,-1,:].sum()}"
    print(updated_logits.shape)
    return


if __name__ == "__main__":
    app.run()
