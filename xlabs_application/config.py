"""Fixed experiment and explicit account budgets (USD, supplied 2026-11-06)."""
MODELS = {
    "base": ["black-forest-labs/FLUX.1-dev", "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"],
    "adapter": ["XLabs-AI/flux-ip-adapter", "18f6940238ab5dc3744df7a8e30315892279d5f9"],
    "clip": ["openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"],
}
SETTINGS = dict(width=1024, height=1024, seed=42, steps=24, guidance_scale=3.5,
                ip_adapter_scale=1.0, strength=1.0, restore_norm=True,
                relative_tolerance=1e-3, mask_single_text_rows=False,
                max_sequence_length=512, true_cfg_scale=1.0)
CONDITIONS = ("baseline", "source_target")
ACCOUNTS = {
    "wj-lee-louis": dict(workspace="wj-lee-louis", credit=7.52, generation_cap=12.0,
                         additional_dreambench_spend_authorized=True,
                         cache="hf-model-cache"),
    "method-comparison": dict(workspace="louislee-85468", credit=7.83, generation_cap=12.0,
                              additional_dreambench_spend_authorized=True,
                              cache="freecus-method-comparison-models"),
}
APP = "xlabs-application-v1"
RUNS = "xlabs-application-runs-v1"
SECRET = "xlabs-application-huggingface"
GPU_RATES = {"L40S": .000542, "A100-80GB": .000694, "H100": .001097, "T4": .000164}
PRICE_DATE = "2026-11-06"


def worker_rate(gpu="L40S"):
    # Conservative billing at 1 physical CPU and the 64 GiB host RAM limit.
    return GPU_RATES[gpu] + .0000131 + 64 * .00000222
