from PIL import Image


def prepare_reference_image(ip_model, reference_image: Image.Image) -> Image.Image:
    from infer_flux_ipa_siglip import resize_img

    return resize_img(reference_image.convert("RGB"))


def apply_conditioning_controls(ip_model, ip_adapter_scale: float):
    ip_model.set_scale(ip_adapter_scale)
    return ip_model
