"""GPU admission based on actual memory and amortized CPU+RAM+GPU cost."""
from freecus.modal_cost import projection


def peak_vram(phase):
    return max(max(t.get("peak_reserved_gib", 0), t.get("peak_allocated_gib", 0))
               for c in phase["calls"] for t in [c["result"].get("initialization") or {}, *c["result"]["timings"]])


def estimate_phase(phase, samples):
    if any(c["result"]["status"] != "complete" or len(c["result"]["timings"]) < 2 for c in phase["calls"]):
        raise ValueError("Successful multi-image pilot required")
    timings = sorted(t["seconds"] for c in phase["calls"] for t in c["result"]["timings"])
    seconds = max(sum(timings) / len(timings), timings[int(0.9 * (len(timings) - 1))])
    overhead = max(c["result"]["elapsed_seconds"] - sum(t["seconds"] for t in c["result"]["timings"]) for c in phase["calls"])
    result = projection(samples, seconds, workers=2, startup_seconds=max(0, overhead), gpu=phase["gpu"], safety=1.25)
    result["peak_vram_gib"] = peak_vram(phase)
    return result


def choose_gpu(estimates):
    if "A100-80GB" not in estimates:
        raise ValueError("A100 baseline is required")
    cheaper = estimates.get("L40S")
    if cheaper and cheaper["peak_vram_gib"] <= 42 and cheaper["compute_usd"] < estimates["A100-80GB"]["compute_usd"] * 0.97:
        return "L40S"
    return "A100-80GB"
