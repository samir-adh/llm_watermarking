import time
import functools
import math
import os
import sys
import random
from typing import Any

import torch
from torch import nn
from dotenv import load_dotenv
from torch import FloatTensor, Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
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
    prompt_len = input_ids.shape[-1]
    attention_mask = inputs["attention_mask"].to(model.device)
    past_key_values = None
    next_token = None

    for _ in range(n_tokens):
        with torch.no_grad():
            model_input_ids = input_ids if past_key_values is None else next_token
            outputs = model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

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
        past_key_values = outputs.past_key_values
    return input_ids[0, prompt_len:]


def prop_under_null(
    tokens: Tensor, green_list: set[int], vocabulary_size: int
) -> float:
    gamma = len(green_list) / vocabulary_size
    n = len(tokens)
    green_tokens = count_green_tokens(tokens, green_list)
    return (
        math.comb(n, green_tokens)
        * ((gamma) ** green_tokens)
        * ((1 - gamma) ** (n - green_tokens))
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
    prompt = "I am currently in the train from "
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    n_tokens = 64
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    assert tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype="bfloat16", token=HF_TOKEN
    )
    vocabulary_size = tokenizer.vocab_size
    gamma = 0.5
    delta = 2
    watermarker = SoftWaterMarker(
        vocabulary_size=vocabulary_size, delta=delta, gamma=gamma, seed=42
    )
    start = time.time()
    output = generate(model, prompt, n_tokens, tokenizer, watermarker)
    ttgen = time.time() - start
    tps = len(output) / ttgen
    print(f"tps={tps}")
    print("output size: ", output.shape[-1])
    green_tokens = count_green_tokens(output, watermarker.green_list)
    z = z_score(green_tokens, len(output), gamma)
    decoded_output = tokenizer.decode(output)
    assert isinstance(decoded_output, str)
    print("prompt and decoded output: \n", prompt + decoded_output)
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
