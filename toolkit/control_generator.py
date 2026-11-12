import gc
import math
import os
import torch
from typing import Literal
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageOps import exif_transpose
from tqdm import tqdm

from torchvision import transforms

# supress all warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def flush(garbage_collect=True):
    torch.cuda.empty_cache()
    if garbage_collect:
        gc.collect()


# conditioning image types (latent-cached in phase 1)
ControlTypes = Literal['depth', 'pose', 'line']
# mask generation types (saved under <image_dir>/_masks/<type>/)
MaskGenTypes = Literal['birefnet', 'inpaint', 'sapiens2']

MASK_GEN_TYPES = ('birefnet', 'inpaint', 'sapiens2')

img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']


def _ensure_local_model_dir(repo_id: str, category_subfolder: str, revision: str = None) -> str:
    """Return a local directory holding a helper model repo's files,
    downloading it on first use. Mask/control models land under
    MODELS_PATH/<category_subfolder>/ -- never the ambient hf cache. A
    completion marker keeps an interrupted download from looking finished."""
    from toolkit.paths import MODELS_PATH

    local_dir = os.path.join(MODELS_PATH, category_subfolder)
    marker = os.path.join(local_dir, '.download_complete')
    if os.path.exists(marker):
        return local_dir
    os.makedirs(local_dir, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
    with open(marker, 'w') as f:
        f.write(repo_id + '\n')
    return local_dir


def generated_image_save_path(img_path: str, gen_type: str) -> str:
    """Where a generated mask/conditioning image lives for a source image.
    Masks go to <image_dir>/_masks/<type>/, conditioning images to
    <image_dir>/_controls/<type>/."""
    img_dir = os.path.dirname(img_path)
    file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
    # masks (single channel) and inpaint (alpha channel) are webp -- far
    # smaller than jpg for both. Conditioning images stay jpg.
    ext = 'webp' if gen_type in MASK_GEN_TYPES else 'jpg'
    if gen_type in MASK_GEN_TYPES:
        folder = os.path.join(img_dir, '_masks', gen_type)
    else:
        folder = os.path.join(img_dir, '_controls', gen_type)
    return os.path.join(folder, f"{file_name_no_ext}.{ext}")


class ControlGenerator:
    def __init__(self, device, sd=None):
        self.device = device
        self.sd = sd  # optional. It will unload the model if not None
        self.has_unloaded = False
        self.control_depth_model = None
        self.control_pose_model = None
        self.control_line_model = None
        # one slot per mask model -- a shared slot let the first-loaded model
        # silently serve every mask type in mixed runs
        self.mask_birefnet_model = None
        self.mask_sapiens2_model = None
        self.debug = False
        self.regen = False

    def get_control_path(self, img_path, gen_type):
        if self.regen:
            return self._generate_control(img_path, gen_type)
        save_path = self.control_save_path(img_path, gen_type)
        if os.path.exists(save_path):
            return save_path
        # if we get here, we need to generate the control
        return self._generate_control(img_path, gen_type)

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def ensure_unloaded(self):
        # park any resident training components (device-state machinery) so
        # the mask/control model loads alone; restored by cleanup()
        if not self.has_unloaded:
            if self.sd is not None:
                print("Unloading training model to generate masks/controls")
                self.sd.set_device_state_preset('unload')
            self.has_unloaded = True

    def load_image(self, img_path):
        # CPU/disk stage: read, orient, and downscale to a max of 1mp
        image = Image.open(img_path).convert('RGB')
        image = exif_transpose(image)

        max_size = 1024 * 1024
        w, h = image.size
        if w * h > max_size:
            scale = math.sqrt(max_size / (w * h))
            w = int(w * scale)
            h = int(h * scale)
            image = image.resize((w, h), Image.BICUBIC)
        return image

    def control_save_path(self, img_path, gen_type):
        return generated_image_save_path(img_path, gen_type)

    def save_control(self, out_image, save_path):
        # CPU/disk stage: encode and write the generated control
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.lower().endswith('.webp'):
            # method=6 trades CPU (already off the GPU thread) for smaller files
            out_image.save(save_path, quality=80, method=6)
        else:
            out_image.save(save_path)

    def _bg_transform(self):
        return transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _ensure_birefnet(self):
        if self.mask_birefnet_model is None:
            from transformers import AutoModelForImageSegmentation

            model_dir = _ensure_local_model_dir(
                'ZhengPeng7/BiRefNet_HR',
                os.path.join('mask', 'birefnet'),
                revision="a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
            )
            print(f"Loading mask model birefnet from {model_dir}")
            self.mask_birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                model_dir,
                trust_remote_code=True,
                dtype=torch.float16
            ).to(self.device)
            self.mask_birefnet_model.eval()

    def preprocess(self, image, gen_type):
        # CPU stage. For the bg-remover path this does the expensive resize +
        # normalize and returns a ready-to-run float16 tensor, so the GPU thread
        # never has to. Other control types preprocess inside their model, so we
        # just pass the PIL image straight through.
        if gen_type in ('birefnet', 'inpaint'):
            return self._bg_transform()(image).unsqueeze(0).to(torch.float16)
        return image

    def run_inference(self, payload, gen_type):
        # GPU stage. Returns an intermediate result for postprocess(). Models are
        # lazily loaded here, so call from a single thread per generator instance.
        self.ensure_unloaded()
        if gen_type in ('birefnet', 'inpaint'):
            self._ensure_birefnet()
            x = payload.to(self.device).to(torch.float16)
            with torch.inference_mode():
                preds = self.mask_birefnet_model(x)[-1].sigmoid().cpu()
            return preds[0].squeeze()  # CPU mask tensor, 1024x1024
        # everything else does preprocessing + inference together on this thread
        return self.run_control(payload, gen_type)

    def postprocess(self, result, image, gen_type):
        # CPU stage. Turns the inference result into the final control image.
        if gen_type in ('birefnet', 'inpaint'):
            mask = transforms.ToPILImage()(result).resize(image.size)
            if gen_type == 'inpaint':
                # inpainting currently only supports the "erased" section to inpaint
                mask = ImageOps.invert(mask)
                out = image.copy()
                out.putalpha(mask)
                return out
            # keep the mask single-channel grayscale; the loader converts as
            # needed and this roughly thirds the file size vs RGB
            return mask
        # the fallback path already produced a finished PIL image
        return result

    def run_control(self, image, gen_type):
        # GPU stage: run inference on an already-loaded image and return the
        # resulting PIL image (no disk IO). Models are lazily loaded here, so
        # this must be called from a single thread per generator instance.
        device = self.device
        self.ensure_unloaded()

        if gen_type == 'depth':
            self.debug_print("Generating depth control")
            if self.control_depth_model is None:
                from transformers import pipeline

                model_dir = _ensure_local_model_dir(
                    'depth-anything/Depth-Anything-V2-Large-hf',
                    os.path.join('control', 'depth-anything-v2'),
                )
                print(f"Loading control model depth (Depth-Anything-V2) from {model_dir}")
                self.control_depth_model = pipeline(
                    task="depth-estimation",
                    model=model_dir,
                    device=device,
                    torch_dtype=torch.float16
                )
            img = image.copy()
            in_size = img.size
            output = self.control_depth_model(img)
            out_tensor = output["predicted_depth"]  # shape (1, H, W) 0 - 255
            out_tensor = out_tensor.clamp(0, 255)
            out_tensor = out_tensor.squeeze(0).cpu().numpy()
            img = Image.fromarray(out_tensor.astype('uint8'))
            img = img.resize(in_size, Image.LANCZOS)
            return img
        elif gen_type == 'pose':
            self.debug_print("Generating pose control")
            if self.control_pose_model is None:
                try:
                    import onnxruntime
                    onnxruntime.set_default_logger_severity(3)
                except ImportError:
                    raise ImportError(
                        "onnxruntime is not installed. Please install it with pip install onnxruntime or onnxruntime-gpu")
                try:
                    from easy_dwpose import DWposeDetector
                    self.control_pose_model = DWposeDetector(
                        device=str(device))
                except ImportError:
                    raise ImportError(
                        "easy-dwpose is not installed. Please install it with pip install git+https://github.com/jaretburkett/easy_dwpose.git")
            img = image.copy()

            detect_res = int(math.sqrt(img.size[0] * img.size[1]))
            img = self.control_pose_model(
                img, output_type="pil", include_hands=True, include_face=True, detect_resolution=detect_res)
            img = img.convert('RGB')
            return img

        elif gen_type == 'line':
            self.debug_print("Generating line control")
            if self.control_line_model is None:
                from controlnet_aux import TEEDdetector

                teed_dir = _ensure_local_model_dir(
                    'fal-ai/teed',
                    os.path.join('control', 'teed'),
                )
                print(f"Loading control model line (TEED) from {teed_dir}")
                self.control_line_model = TEEDdetector.from_pretrained(
                    teed_dir, filename="5_model.pth").to(device)
            img = image.copy()
            img = self.control_line_model(img, detect_resolution=1024)
            # apply threshold
            # img = img.filter(ImageFilter.GaussianBlur(radius=1))
            img = img.point(lambda p: p > 128 and 255)
            img = img.convert('RGB')
            return img
        elif gen_type in ('birefnet', 'inpaint', 'sapiens2'):
            self.debug_print(f"Generating {gen_type} mask")
            if gen_type == 'sapiens2':
                if self.mask_sapiens2_model is None:
                    from toolkit.models.sapiens2 import Sapiens2Matting
                    self.mask_sapiens2_model = Sapiens2Matting.from_pretrained(
                        device=device,
                        dtype=torch.float16
                    )
                img = image.copy()
                img = self.mask_sapiens2_model(img)
                return img
            # birefnet and inpaint share the bg-remover model and differ only
            # in postprocess packaging
            payload = self.preprocess(image, gen_type)
            result = self.run_inference(payload, gen_type)
            return self.postprocess(result, image, gen_type)
        else:
            raise Exception(f"Error: unknown control type {gen_type}")

    def _generate_control(self, img_path, control_type):
        image = self.load_image(img_path)
        out_image = self.run_control(image, control_type)
        save_path = self.control_save_path(img_path, control_type)
        self.save_control(out_image, save_path)
        return save_path

    def cleanup(self):
        if self.control_depth_model is not None:
            self.control_depth_model = None
        if self.control_pose_model is not None:
            self.control_pose_model = None
        if self.control_line_model is not None:
            self.control_line_model = None
        if self.mask_birefnet_model is not None:
            self.mask_birefnet_model = None
        if self.mask_sapiens2_model is not None:
            self.mask_sapiens2_model = None
        if self.sd is not None and self.has_unloaded:
            self.sd.restore_device_state()
        self.has_unloaded = False

        flush()


if __name__ == "__main__":
    import sys
    import argparse
    import time
    import transformers
    transformers.logging.set_verbosity_error()

    control_times = {
        'depth': 0,
        'pose': 0,
        'line': 0,
        'birefnet': 0,
        'inpaint': 0,
        'sapiens2': 0
    }

    controls = control_times.keys()

    parser = argparse.ArgumentParser(description="Generate control images")
    parser.add_argument("img_dir", type=str, help="Path to image directory")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode")
    parser.add_argument('--regen', action='store_true',
                        help="Regenerate all controls")

    args = parser.parse_args()
    img_dir = args.img_dir
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} does not exist")
        exit()
    if not os.path.isdir(img_dir):
        print(f"Error: {img_dir} is not a directory")
        exit()

    # find images
    img_list = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if "_controls" in root or "_masks" in root or "_latent_cache" in root:
                continue
            if file.startswith('.'):
                continue
            if file.lower().endswith(tuple(img_ext_list)):
                img_list.append(os.path.join(root, file))
    if len(img_list) == 0:
        print(f"Error: no images found in {img_dir}")
        exit()

    # load model
    idx = 0
    for img_path in tqdm(img_list):
        for control in controls:
            start = time.time()
            control_gen = ControlGenerator(torch.device('cuda'))
            control_gen.debug = args.debug
            control_gen.regen = args.regen
            control_path = control_gen.get_control_path(img_path, control)
            end = time.time()
            # dont track for first 2 images
            if idx < 2:
                continue
            control_times[control] += end - start
        idx += 1

    # determine avgt time
    for control in controls:
        control_times[control] /= (idx - 2)
        print(
            f"Avg time for {control} control: {control_times[control]:.2f} seconds")

    print("Done")
