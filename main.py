import os
import json
import random

import pandas as pd
import torch

from transformer_lens import HookedTransformer
from sae_lens import SAE
from dotenv import load_dotenv


os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
load_dotenv()

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

MODEL_NAME = "google/gemma-2-2b"

model = HookedTransformer.from_pretrained_no_processing(
    MODEL_NAME,
    device=DEVICE,
    dtype=DTYPE,
)

sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device=DEVICE,
)

sae = sae.to(dtype=DTYPE)

# Residual-stream SAE for layer 12
hook_name = "blocks.12.hook_resid_post"


@torch.inference_mode()
def decompose(prompt):
    tokens = model.to_tokens(prompt)

    logits, cache = model.run_with_cache(
        tokens,
        names_filter=[hook_name],
    )

    x = cache[hook_name][:, -1, :]
    x_hat = sae.decode(sae.encode(x))
    epsilon = x - x_hat

    return {
        "x_hat": x_hat.float().cpu(),
        "epsilon": epsilon.float().cpu(),
        "logits": logits[:, -1].float().cpu(),
    }


if __name__ == "__main__":
    INPUT_CSV = "data/TruthfulQA.csv"
    OUTPUT_CSV = "data/truthfulqa_activations.csv"

    df = pd.read_csv(INPUT_CSV)

    token_a = model.to_single_token(" A")
    token_b = model.to_single_token(" B")

    rng = random.Random(42)
    results = []

    for _, row in df.iterrows():
        correct_answer = row["Best Answer"]
        incorrect_answer = row["Best Incorrect Answer"]

        # Randomize answer positions
        if rng.random() < 0.5:
            option_a = correct_answer
            option_b = incorrect_answer
            correct_option = "A"
        else:
            option_a = incorrect_answer
            option_b = correct_answer
            correct_option = "B"

        prompt = (
            f"Question: {row['Question']}\n"
            f"A: {option_a}\n"
            f"B: {option_b}\n"
            "Answer:"
        )

        result = decompose(prompt)

        logits = result["logits"].squeeze(0)
        logit_a = logits[token_a].item()
        logit_b = logits[token_b].item()

        predicted = "A" if logit_a > logit_b else "B"
        hallucination = int(predicted != correct_option)

        confidence = torch.softmax(
            torch.tensor([logit_a, logit_b]),
            dim=0,
        ).max().item()

        results.append({
            **row.to_dict(),
            "Prompt": prompt,
            "Option A": option_a,
            "Option B": option_b,
            "Correct Option": correct_option,
            "Predicted Option": predicted,
            "Hallucination Label": hallucination,
            "Confidence": confidence,
            "SAE Hook": hook_name,
            "x_hat": json.dumps(
                result["x_hat"].squeeze(0).tolist()
            ),
            "epsilon": json.dumps(
                result["epsilon"].squeeze(0).tolist()
            ),
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(results)} rows to {OUTPUT_CSV}")
