from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    FineGrainedFP8Config,
)
import subprocess, sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "mistral-common"], check=True
)
import kagglehub

model_id = kagglehub.model_download(
    "mistral-ai/ministral-3/Transformers/ministral-3-3b-base-2512"
)


model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
)
tokenizer = MistralCommonBackend.from_pretrained(model_id)

input_ids = tokenizer.encode("Once about a time, France was a", return_tensors="pt")
input_ids = input_ids.to("cuda")

output = model.generate(
    input_ids,
    max_new_tokens=30,
)[0]

decoded_output = tokenizer.decode(output[len(input_ids[0]) :])
print(decoded_output)
