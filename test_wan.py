import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/palakons/Wan2.2")
from wan.modules.vae2_1 import Wan2_1_VAE


def test_wanvae_frame_count(
    frame_counts=(1, 2, 3, 4, 5, 9),
    image_hw=(943, 1980),
    resized_hw=(480, 832),
    device="cuda",
    wan_vae_checkpoint="/checkpoints/huggingface_hub/models--Wan-AI--Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
):
    print("Loading Wan2.1 VAE...")
    vae = Wan2_1_VAE(
        vae_pth=wan_vae_checkpoint,
        device=device,
    )
    # vae.eval()

    H0, W0 = image_hw
    h, w = resized_hw

    # Fake RGB image in [-1, 1], same as your preprocessing output
    img = torch.rand(1, 3, H0, W0, device=device) * 2.0 - 1.0

    # Resize to your VAE input size
    img = F.interpolate(
        img,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )  # [1, 3, h, w]

    results = {}

    for num_frames in frame_counts:
        print("\n" + "=" * 60)
        print(f"Testing F = {num_frames}")

        # [1, 3, h, w] -> [1, 3, F, h, w]
        videos = img.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)

        print("input video shape:", tuple(videos.shape))
        print("input min/max:", videos.min().item(), videos.max().item())

        try:
            with torch.inference_mode():
                latent = vae.encode(videos)[0]

            print("ENCODE OK")
            print("latent shape:", tuple(latent.shape))
            print("latent dtype:", latent.dtype)
            print("latent min/max:", latent.min().item(), latent.max().item())

            results[num_frames] = tuple(latent.shape)

        except Exception as e:
            print("ENCODE FAILED")
            print(type(e).__name__)
            print(str(e))

            results[num_frames] = None

    print("\n" + "=" * 60)
    print("Summary:")
    for num_frames, shape in results.items():
        print(f"F={num_frames}: {shape}")

    return results


if __name__ == "__main__":
    test_wanvae_frame_count()