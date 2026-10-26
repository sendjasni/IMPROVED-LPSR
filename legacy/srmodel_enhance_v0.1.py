#!/usr/bin/env python3
"""
A Real-ESRGAN x2 based License Plate Enhancer
================================================
Place this script and the file 'weights_x2.pth' in the same folder.
No internet connection required after the first setup.

Usage:
    python srmodel_enhance_v0.1.py --input plate.jpg
    python srmodel_enhance_v0.1.py --input ./plates --output ./enhanced

Requirements (install once):
    pip install torch realesrgan opencv-python pillow tqdm
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# -------------------------------------------------------------------
# 1. Locate the weights file
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = SCRIPT_DIR / "weights_x2.pth"

def get_weights_path():
    """
    Returns the path to the model weights.
    If not found locally, it offers to download them automatically.
    """
    if WEIGHTS_FILE.exists():
        return str(WEIGHTS_FILE)

    # If missing, ask user if they want to download
    print("=" * 60)
    print("Model weights not found in the script folder.")
    print(f"Expected location: {WEIGHTS_FILE}")
    answer = input("Download them now? (~67 MB) [Y/n]: ").strip().lower()
    if answer and answer != 'y':
        print("Aborting. Please place 'weights_x2.pth' next to the script.")
        sys.exit(1)


# -------------------------------------------------------------------
# 2. Real-ESRGAN x2 upsampler
# -------------------------------------------------------------------
def load_esrgan_x2(device="cuda"):
    """Load the Real-ESRGAN x2 model and return an upsampler object."""
    import torch
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    weights_path = get_weights_path()

    # Try the newer realesrgan package first
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(
            scale=2,
            model_path=weights_path,
            model=model,
            tile=0,           # process whole image (plates are small)
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device,
        )
        return upsampler
    except ImportError:
        pass

    # Fallback: use the realesrgan package (pip install realesrgan)
    try:
        from realesrgan import RealESRGAN as RealESRGANCls
        model = RealESRGANCls(device=device, scale=2)
        model.load_weights(weights_path, download=False)
        class ESRGANWrapper:
            def __init__(self, model):
                self.model = model
            def enhance(self, img_bgr, outscale=2):
                from PIL import Image
                pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                sr_pil = self.model.predict(pil)
                return cv2.cvtColor(np.array(sr_pil), cv2.COLOR_RGB2BGR), None
        return ESRGANWrapper(model)
    except ImportError:
        print("Please install realesrgan or basicsr: pip install realesrgan")
        sys.exit(1)


def enhance_image(upsampler, img_bgr):
    """Run 2x enhancement on a BGR image. Returns BGR image."""
    output, _ = upsampler.enhance(img_bgr, outscale=2)
    return output


# -------------------------------------------------------------------
# 3. Command‑line interface
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN x2 License Plate Enhancer")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default=None, help="Output file (single image) or output directory (batch)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    input_path = Path(args.input)
    upsampler = load_esrgan_x2(args.device)

    # ----- Single image -----
    if input_path.is_file():
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Cannot read {input_path}")
            sys.exit(1)
        print(f"Enhancing {input_path.name} ({img.shape[1]}x{img.shape[0]}) …")
        enhanced = enhance_image(upsampler, img)
        out_file = args.output or input_path.with_name(input_path.stem + "_enhanced.png")
        cv2.imwrite(str(out_file), enhanced)
        print(f"Saved → {out_file}")
        return

    # ----- Directory of images -----
    if input_path.is_dir():
        out_dir = Path(args.output) if args.output else input_path / "enhanced"
        out_dir.mkdir(parents=True, exist_ok=True)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [p for p in input_path.iterdir() if p.suffix.lower() in extensions]
        if not files:
            print("No images found.")
            return
        print(f"Processing {len(files)} images …")
        for p in tqdm(files, desc="Real-ESRGAN x2"):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            enhanced = enhance_image(upsampler, img)
            cv2.imwrite(str(out_dir / p.name), enhanced)
        print(f"All enhanced images saved to {out_dir}")
        return

    print("Input must be a file or directory.")
    sys.exit(1)


if __name__ == "__main__":
    main()