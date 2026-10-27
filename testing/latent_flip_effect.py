"""Latent cell-shuffle flip filter (an artistic effect, not a real flip).

A plain torch.flip of a channel-PACKED cached latent (the FLUX.2-style
AE families: moss/flux2/flux3/fibo2/ideogram4/ernie) reverses the
16-pixel latent cells but leaves the in-cell channel content unmapped.
The VAE decoder faithfully reconstructs that state: rows/columns land
back in the mirrored order, yet each band's content comes out
upright-ish and subtly wrong -- the VAE "heals" the mismatch into an
optical illusion. A correct flipped latent must be ENCODED from
flipped pixels (training uses that path); the effect here is what
happens when you flip the latent instead, kept as a deliberate filter.

The illusion is strongest on packed-AE models. --res controls the
bucket area: every latent cell spans 16 pixels at scale, so lowering
--res coarsens the effect into larger bands.

Examples:
    venv/bin/python testing/latent_flip_effect.py input/photo.jpg
    venv/bin/python testing/latent_flip_effect.py input/*.jpg --res 256 --side_by_side
    venv/bin/python testing/latent_flip_effect.py photo.jpg --modes flipx flipy both
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

from toolkit.config_modules import ModelConfig
from toolkit.util.get_model import get_model_class

MODES = {
    'flipx': [-1],  # mirror the latent grid horizontally
    'flipy': [-2],  # mirror the latent grid vertically
    'both': [-1, -2],
}


def scaled_grid_size(w, h, area, div):
    # px-area override that may UPSCALE (the shared bucket helper never
    # enlarges, being training geometry): scale to area, round to grid
    scaler = (area / (w * h)) ** 0.5
    w2 = max(div, int(round(w * scaler / div)) * div)
    h2 = max(div, int(round(h * scaler / div)) * div)
    return h2, w2


def to_pil(t):
    t = (t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5
    return Image.fromarray(t.permute(1, 2, 0).round().to(torch.uint8).numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('images', nargs='+', help='input image file(s)')
    ap.add_argument('--out_dir', default=None,
                    help='output directory (default: alongside each input)')
    ap.add_argument('--arch', default='moss_image',
                    help='model whose VAE decodes the effect (packed-AE '
                         'archs give the illusion; conv VAEs flip almost '
                         'cleanly and show almost nothing)')
    ap.add_argument('--name_or_path', default=None,
                    help='override the model location (default: the arch '
                         'resolver finds the shared VAE)')
    ap.add_argument('--res', type=int, default=512,
                    help='pixel-area override: encode at an area of '
                         'res*res px instead of the model\'s usual training '
                         'bucket (input is resized to the model-grid bucket '
                         'of that area). 0 = keep the input size, floored to '
                         'the latent grid. Every latent cell spans 16 px, so '
                         'smaller res = coarser illusion bands. Any input '
                         'size/aspect works')
    ap.add_argument('--modes', nargs='+', choices=list(MODES),
                    default=['flipx', 'flipy'],
                    help='which latent mirror(s) to render')
    ap.add_argument('--roundtrip', action='store_true',
                    help='also save the plain encode->decode reconstruction')
    ap.add_argument('--side_by_side', action='store_true',
                    help='also save original | effect grids')
    args = ap.parse_args()

    missing = [p for p in args.images if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f'file(s) not found: {missing}')

    cfg_kwargs = {'arch': args.arch}
    name_or_path = args.name_or_path
    if not name_or_path:
        # borrow the arch's own default repo instead of hardcoding one
        try:
            import importlib
            mod = importlib.import_module(
                f'extensions_built_in.diffusion_models.{args.arch}.src.transformer'
            )
            name_or_path = getattr(mod, 'VENDOR_BASE_REPO', None)
        except Exception:
            pass
    if not name_or_path:
        raise SystemExit(
            f'--name_or_path is required for arch {args.arch!r} '
            f'(no module default found)'
        )
    cfg_kwargs['name_or_path'] = name_or_path
    model_config = ModelConfig(**cfg_kwargs)
    model_class = get_model_class(model_config)
    sd = model_class(device='cuda', model_config=model_config,
                    dtype=torch.bfloat16, custom_pipeline=None,
                    noise_scheduler=None)
    # the phase hook loads this model's own VAE profile through the
    # resolver chain; the transformer/text encoder are never touched
    sd.load_vae()

    div = sd.get_bucket_divisibility()
    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for path in args.images:
        stem = os.path.splitext(os.path.basename(path))[0]
        dst = out_dir or os.path.dirname(os.path.abspath(path))
        img = Image.open(path).convert('RGB')
        w, h = img.size
        if args.res > 0:
            size = scaled_grid_size(w, h, args.res * args.res, div)
        else:
            size = (h // div * div or div, w // div * div or div)
        x = TF.resize(TF.to_tensor(img), size, antialias=True).cuda() * 2 - 1
        with torch.no_grad():
            z = sd.encode_images([x])
            tiles = {'original': to_pil(x)}
            outs = {}
            if args.roundtrip:
                outs['roundtrip'] = to_pil(sd.decode_latents(z)[0])
            for mode in args.modes:
                zf = torch.flip(z, MODES[mode])
                outs[mode] = to_pil(sd.decode_latents(zf)[0])
        for name, pil in outs.items():
            out_path = os.path.join(dst, f'{stem}_latent_{name}.png')
            pil.save(out_path)
            print(f'{out_path}')
        if args.side_by_side:
            for mode in args.modes:
                left, right = tiles['original'], outs[mode]
                tw = max(left.width, right.width)
                th = max(left.height, right.height)
                pad = 18
                grid = Image.new('RGB', (tw * 2 + 12, th + pad + 8), (24, 24, 24))
                d = ImageDraw.Draw(grid)
                grid.paste(left, (4, 4))
                grid.paste(right, (tw + 8, 4))
                d.text((4, th + 2), 'original', fill=(230, 230, 230))
                d.text((tw + 8, th + 2), f'latent {mode}', fill=(230, 230, 230))
                out_path = os.path.join(dst, f'{stem}_latent_{mode}_grid.png')
                grid.save(out_path)
                print(f'{out_path}')


if __name__ == '__main__':
    main()
