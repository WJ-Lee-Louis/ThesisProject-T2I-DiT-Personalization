import unittest
import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionSdpaAttention
from freecus.runtime_src.qwen_for_subject import vision_sdpa_without_dense_mask


class QwenVisionAttentionTests(unittest.TestCase):
    def test_matches_dense_mask_for_multiple_images_and_rotary_embeddings(self):
        torch.manual_seed(42)
        layer = VisionSdpaAttention(32, num_heads=4).eval()
        hidden = torch.randn(13, 32)
        offsets = torch.tensor([0, 5, 13], dtype=torch.int32)
        angles = torch.randn(13, 8)
        embeddings = (angles.cos(), angles.sin())
        with torch.inference_mode():
            expected = layer(hidden, offsets, position_embeddings=embeddings)
            actual = vision_sdpa_without_dense_mask(layer, hidden, offsets, position_embeddings=embeddings)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
