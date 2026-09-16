import dotenv
import os
import torch

from tqdm.auto import tqdm
# from transformer_lens import HookedTransformer
from sae_lens import SAE

from dotenv import load_dotenv
# from transformers import AutoTokenizer, GPTNeoXForCausalLM

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1" 
device = "mps" if torch.backends.mps.is_available() else "cpu"

load_dotenv()

def load_sae():

    sae, cfg_dict, sparsify = SAE.from_pretrained_with_cfg_and_sparsity(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_16k/canonical",
        device=device,
    )

    # activations shape: (batch, seq_len, d_model)
    # Gemma 2 2B has d_model of 2304
    activations = torch.randn(1, 128, 2304, device=device)

    # feature_acts shape: (batch, seq_len, d_sae)
    features = sae.encode(activations)

    # Check which features are active
    active_features = (features > 0).sum(dim=-1)

    reconstructed = sae.decode(features)

    print(activations)
    print("="*75)
    print(features)
    print("="*75)
    print(f"Average L0: {active_features.float().mean().item()}")
    print("="*75)
    print(reconstructed)

    mse = (activations - reconstructed).pow(2).mean()
    print(f"Reconstruction Error: {mse.item()}")

if __name__ == "__main__":
    load_sae()
