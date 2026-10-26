# enhance_sequence.py
"""
LPSR v0.2 — amélioration multi-images de plaques d'immatriculation (super-résolution x2 ou x4).

Entrée : un dossier contenant une séquence d'images d'un même véhicule (idéalement 10), soit
  (a) les images complètes + detections.json  [{"frame": "000000.png", "license_plate_coordinates": [x1, y1, x2, y2]}, ...]
  (b) uniquement des recadrages de la plaque, avec une marge autour de la plaque
      (plaque ≈ 2/3 central en largeur, 1/2 central en hauteur, comme le produit le mode (a)).
Sortie (dossier --output) :
  plaque_XX_originale.png, plaque_XX_bicubique_xS.png, plaque_XX_amelioree_xS.png   (une par image)
  planche.png   planche comparative, marquée « image améliorée : aide visuelle, ne constitue pas une preuve »

Usage :
  python enhance_sequence.py --input dossier_sequence --output resultats            # x2
  python enhance_sequence.py --input dossier_sequence --output resultats --scale 4  # x4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lpsr.model import MFSR

MARGIN_X, MARGIN_Y = 0.25, 0.5          # marge ajoutée autour de la boîte plaque (fraction largeur / hauteur)
WARNING = ("Image améliorée : aide visuelle à l'enquête, ne constitue pas une preuve. "
           "Se référer aux images originales.")


def crop_plate(img, box):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = max(0, round(x1 - MARGIN_X * w)), max(0, round(y1 - MARGIN_Y * h))
    cx2, cy2 = min(W, round(x2 + MARGIN_X * w)), min(H, round(y2 + MARGIN_Y * h))
    return img[cy1:cy2, cx1:cx2]


def load_sequence(folder):
    """Returns [(name, crop)]. Crops are cut with the same margins as during training."""
    folder = Path(folder)
    det = folder / 'detections.json'
    if det.exists():
        return [(d['frame'], crop_plate(cv2.imread(str(folder / d['frame'])), [int(v) for v in d['license_plate_coordinates']]))
                for d in json.load(open(det))]
    return [(p.name, cv2.imread(str(p))) for p in sorted(folder.glob('*'))
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]


def to_tensor(frames, device):
    x = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).transpose(0, 3, 1, 2).astype(np.float32) / 255
    return torch.from_numpy(x)[None].to(device)


@torch.no_grad()
def enhance(model, crops, device):
    """Each frame in turn is the reference; the other frames are resized onto it (same margins)."""
    out = []
    for ref in range(len(crops)):
        h, w = crops[ref].shape[:2]
        seq = [crops[ref]] + [cv2.resize(crops[i], (w, h), interpolation=cv2.INTER_CUBIC)
                              for i in range(len(crops)) if i != ref]
        y = model(to_tensor(seq, device))[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        out.append(cv2.cvtColor((y * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
    return out


def contact_sheet(originals, bicubic, enhanced, scale, path):
    H = 100
    rs = lambda im: cv2.resize(im, (round(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_CUBIC)
    rows = [[rs(o), rs(b), rs(e)] for o, b, e in zip(originals, bicubic, enhanced)]
    col_w = [max(max(r[k].shape[1] for r in rows), 290) for k in range(3)]
    pad = lambda im, wd: cv2.copyMakeBorder(im, 0, 0, 0, wd - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    body = np.vstack([np.hstack([pad(r[k], col_w[k] + 10) for k in range(3)]) for r in rows])
    head_h = 80
    sheet = Image.new('RGB', (body.shape[1], body.shape[0] + head_h), 'white')
    sheet.paste(Image.fromarray(cv2.cvtColor(body, cv2.COLOR_BGR2RGB)), (0, head_h))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    first, second = WARNING.split('. ', 1)
    draw.text((6, 4), first + '.', fill=(170, 0, 0), font=font)
    draw.text((6, 24), second, fill=(170, 0, 0), font=font)
    x = 0
    for title, wd in zip(['Originale', f'Bicubique x{scale}', f'Améliorée x{scale} (multi-images)'], col_w):
        draw.text((x + 6, 56), title, fill=(0, 0, 0), font=font)
        x += wd + 10
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--scale', type=int, default=2, choices=[2, 4])
    ap.add_argument('--weights', default=None, help='par défaut : weights_lpsr_v0.2_x2.pth / _x4.pth selon --scale')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no_sheet', action='store_true', help='ne pas produire la planche comparative')
    args = ap.parse_args()

    weights = args.weights or str(Path(__file__).parent / f'weights_lpsr_v0.2_x{args.scale}.pth')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seq = load_sequence(args.input)
    if not seq:
        raise SystemExit(f'Aucune image trouvée dans {args.input}')
    names, crops = zip(*seq)

    state = torch.load(weights, map_location=args.device)
    model = MFSR(**state['model_cfg']).to(args.device).eval()
    model.load_state_dict(state['mfsr'])
    if model.scale != args.scale:
        raise SystemExit(f'{weights} est un modèle x{model.scale}, pas x{args.scale}')

    enhanced = enhance(model, list(crops), args.device)
    bicubic = [cv2.resize(c, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC) for c in crops]
    for k, (c, b, e) in enumerate(zip(crops, bicubic, enhanced)):
        cv2.imwrite(str(out / f'plaque_{k:02d}_originale.png'), c)
        cv2.imwrite(str(out / f'plaque_{k:02d}_bicubique_x{args.scale}.png'), b)
        cv2.imwrite(str(out / f'plaque_{k:02d}_amelioree_x{args.scale}.png'), e)
    if not args.no_sheet:
        contact_sheet(crops, bicubic, enhanced, args.scale, out / 'planche.png')
    print(f'{len(names)} images améliorées (x{args.scale}) -> {out}')
    print(WARNING)


if __name__ == '__main__':
    main()
