import random

import torch
from torch import  Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


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


def generate(
    model: PreTrainedModel,
    model_name: str,
    prompt: str,
    n_tokens: int,
    watermarker: SoftWaterMarker | None = None,
) -> str:
    tokenizer= AutoTokenizer.from_pretrained(model_name)
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
        next_token_logits = logits[:,-1,:]
        probs = torch.softmax(next_token_logits,dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token],dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    model_name= "LiquidAI/LFM2.5-350M"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    prompt = "hello"
    n_tokens = 1
    watermarker = None
    output = generate(model, model_name, prompt, n_tokens)
    print(output)



if __name__ == "__main__":
    main()
