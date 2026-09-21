import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


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
    import random
    import numpy as np
    from torch import FloatTensor, Tensor

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
                new_logits[:, -1, i] += self.delta
            return new_logits

    return FloatTensor, SoftWaterMarker, Tensor


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
    return AutoTokenizer, logits, model, model_name, torch


@app.cell
def _(SoftWaterMarker, logits: "FloatTensor", torch):
    vocabulary_size = logits.shape[2]
    watermaker = SoftWaterMarker(
        seed=42, vocabulary_size=vocabulary_size, ratio=0.5, delta=2
    )
    updated_logits = watermaker.update_logits(
        logits
    )  # here we could optimize by taking only the last logits
    updated_logits = torch.softmax(updated_logits, dim=2)
    assert updated_logits[:, -1, :].sum() == 1.0, (
        f"sum of logits should be 1 but got {updated_logits[:, -1, :].sum()}"
    )
    print(updated_logits.shape)
    return


@app.cell
def _(AutoTokenizer, SoftWaterMarker, Tensor, torch):
    from transformers import PreTrainedModel

    def generate(
        model: PreTrainedModel,
        model_name: str,
        prompt: str,
        n_tokens: int,
        watermarker: SoftWaterMarker | None = None,
    ) -> list[str]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert tokenizer
        inputs: dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        for _ in range(n_tokens):
            with torch.no_grad():
                outputs = model(**inputs)

            logits: Tensor = outputs.logits
            if watermarker:
                logits = watermarker.update_logits(logits)
                assert logits[:, -1, :].sum() == 1.0, (
                    f"sum of logits should be 1 but got {logits[:, -1, :].sum()}"
                )
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )
            if next_token.item() == tokenizer.eos_token_id:
                break
            print("next token: ", tokenizer.decode(next_token))
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return (generate,)


@app.cell
def _(generate, model, model_name):
    prompt = """
    Alice: Hello Bob, how are you?
    Bob: Hello Alice, I'm fine what"""
    n_tokens = 2
    watermarker = None
    output = generate(model, model_name, prompt, n_tokens, watermarker)
    print(output)
    return


if __name__ == "__main__":
    app.run()
