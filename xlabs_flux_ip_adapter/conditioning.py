from PIL import Image


def prepare_reference_image(reference_image: Image.Image, width: int, height: int) -> Image.Image:
    return reference_image.convert("RGB").resize((width, height))


def apply_conditioning_controls(pipe, ip_adapter_scale: float):
    """Central hook for future training-free conditioning interventions."""
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    return pipe
