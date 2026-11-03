"""Transparent Modal estimates; measured runtimes are required for admission."""
import argparse
import json

RATES = {"A100-80GB": 0.000694, "H100": 0.001097, "L40S": 0.000542}
CPU_SECOND = 0.0000131
GIB_SECOND = 0.00000222
PRICE_DATE = "2026-10-30"


def rate(gpu="A100-80GB", cpu=2, memory_gib=64):
    return RATES[gpu] + cpu * CPU_SECOND + memory_gib * GIB_SECOND


def projection(samples, seconds_per_image, workers=2, startup_seconds=0, gpu="A100-80GB", safety=1.25):
    gpu_seconds = samples * seconds_per_image + workers * startup_seconds
    usd = gpu_seconds * rate(gpu)
    return dict(samples=samples, workers=workers, gpu=gpu, seconds_per_image=seconds_per_image,
                total_worker_seconds=gpu_seconds, wall_seconds=gpu_seconds / workers,
                compute_usd=usd, compute_usd_with_margin=usd * safety,
                marginal_usd_per_image=seconds_per_image * rate(gpu), safety_factor=safety)


def report(seconds=30, startup=120):
    return dict(price_date=PRICE_DATE, source="https://modal.com/pricing", cpu_cores=2, host_memory_gib=64,
                hourly_worker_usd=rate() * 3600, budget_usd=30, reserve_usd=5,
                optimistic_seconds_per_image_limit_25usd={"dreambench": 25 / 750 / rate(), "both": 25 / 2100 / rate()},
                scenarios=[projection(n, seconds, startup_seconds=startup) for n in [750, 1350, 2100]],
                note="Runtime scenarios are estimates, not measured speed. CPU/RAM actual usage above requests, builds, retries and other workspace jobs can add charges.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds-per-image", type=float, default=30)
    p.add_argument("--startup-seconds", type=float, default=120)
    a = p.parse_args()
    print(json.dumps(report(a.seconds_per_image, a.startup_seconds), indent=2))


if __name__ == "__main__":
    main()
