import functools
import math
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch import FloatTensor, Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TokenizersBackend,
)
from transformers.generation.watermarking import WatermarkDetector
from transformers.modeling_outputs import CausalLMOutputWithPast

load_dotenv()

MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HUGGING_FACE_TOKEN"]
# MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Base"
VOCABULARY_SIZE = 50272


class MockLLM(nn.Module):
    def __init__(self, vocabulary_size=VOCABULARY_SIZE) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocabulary_size, 1)
        self.lm_head = nn.Linear(1, vocabulary_size)
        self.vocabulary_size = vocabulary_size

    @property
    def device(self) -> torch.device:
        return self.embeddings.weight.device

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs: Any
    ) -> CausalLMOutputWithPast:
        # input_ids: (batch, n_tokens) -> logits: (batch, n_tokens, vocab_size)
        output = FloatTensor(
            torch.randn(size=(*input_ids.size(), self.vocabulary_size))
        )
        return CausalLMOutputWithPast(logits=output)


class SoftWaterMarker:
    def __init__(
        self, seed: int | None, vocabulary_size: int, gamma: float, delta: float
    ) -> None:
        self.seed = seed
        self.delta = delta
        generator = random.Random(seed)
        shuffled_indices = [i for i in range(vocabulary_size)]
        generator.shuffle(shuffled_indices)
        green_list_size = int(vocabulary_size * gamma)
        # For now we create the green list in advanced.
        # TODO: use the hash of the token t_0 to seed the generator at each token.
        self.green_list = set(shuffled_indices[:green_list_size])
        self.green_idx = torch.tensor(sorted(self.green_list))
        # self.red_list = shuffled_indices[green_list_size:]

    def update_probs(self, logits: Tensor) -> Tensor:
        """
        Updates the logits of the green list by adding delta to their value.

        Args:
            logits (list[float]): Logits predicted by the model.

        Returns:
            tuple[list[float], list[int], list[int]]: Updated logits, green list and red list.
        """
        new_logits: Tensor = logits.clone()
        new_logits[:, -1, self.green_idx.to(logits.device)] += self.delta
        return new_logits


def generate(
    model: PreTrainedModel,
    prompt: str,
    n_tokens: int,
    tokenizer: Any,
    watermarker: SoftWaterMarker | None = None,
) -> Tensor:
    inputs: dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    for _ in range(n_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits: Tensor = outputs.logits
        if watermarker:
            logits = watermarker.update_probs(logits)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat(
            [input_ids, next_token], dim=-1
        )  # size = (batch, k_tokens)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=-1
        )
        if next_token.item() == tokenizer.eos_token_id:
            break
        # print("next token: ", tokenizer.decode(next_token))
    return input_ids[0]


def binomial(k: int, n: int):
    assert k <= n

    @functools.cache
    def factorial(n: int):
        assert n >= 0
        if n == 0:
            return 1
        else:
            return n * factorial(n - 1)

    return factorial(n) / (factorial(k) * factorial(n - k))


def prop_under_null(
    tokens: Tensor, green_list: set[int], vocabulary_size: int
) -> float:
    gamma = len(green_list) / vocabulary_size
    n = len(tokens)
    green_tokens = count_green_tokens(tokens, green_list)
    return (
        math.comb(n, green_tokens)
        * ((gamma) ** green_tokens)
        * ((gamma) ** (n - green_tokens))
    )


def z_score(green_tokens_count: int, n_tokens: int, gamma: float) -> float:
    output = (green_tokens_count - gamma * n_tokens) / math.sqrt(
        n_tokens * gamma * (1 - gamma)
    )
    return output


def count_green_tokens(output: Tensor, green_set: set[int]):
    count = 0
    for token in output:
        if int(token) in green_set:
            count += 1
    return count


def main():
    prompt = "hello"
    n_tokens = 100
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    assert tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype="bfloat16",token=HF_TOKEN
    )
    # model = MockLLM()
    vocabulary_size = VOCABULARY_SIZE
    gamma = 0.5
    delta = 2
    watermarker = SoftWaterMarker(
        vocabulary_size=vocabulary_size, delta=delta, gamma=gamma, seed=42
    )
    output = generate(model, prompt, n_tokens, tokenizer, watermarker)
    print("output size: ", output.size())
    green_tokens = count_green_tokens(output, watermarker.green_list)
    z = z_score(green_tokens, len(output), gamma)
    print("green list:", list(watermarker.green_list)[:5])
    print("output: ", output[:5])
    decoded_output = tokenizer.decode(output)
    print("decoded output: ", decoded_output)
    print("z=", z)


def test_tokenizer():
    prompt = "will this ever bee tokenized the same way ????"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    assert tokenizer
    prev = tokenizer.encode(prompt)
    for i in range(10):
        current = tokenizer.encode(prompt)
        assert prev == current, (
            f"len(prev)=={len(prev)} vs len(current)=={len(current)}"
        )
        prev = current
    print(len(prev))


if __name__ == "__main__":
    main()
