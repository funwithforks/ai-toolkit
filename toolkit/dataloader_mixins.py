import base64
import glob
import hashlib
import itertools
import json
import math
import os
import random
import time
from collections import OrderedDict, deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Dict, Union

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor

from toolkit.audio.preserve_pitch import time_stretch_preserve_pitch
from toolkit.basic import UnusableFileError, flush, get_quick_signature_string, value_map
from toolkit.buckets import get_bucket_for_image_size, get_resolution
from toolkit.config_modules import ControlTypes, MASK_MODEL_PASS_ORDER, MASK_MODEL_VALUES
from toolkit.control_generator import ControlGenerator, generated_image_save_path
from toolkit.dto import DTO, DISK_PREFIX
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.pixtral_vision import PixtralVisionImagePreprocessorCompatible
from toolkit.prompt_utils import inject_trigger_into_prompt, extract_inline_dop_pairs, apply_inline_dop
from toolkit.video_ingest import (
    compute_auto_frame_count,
    load_clip_audio,
    load_dataset_float_tensors,
)
from toolkit.video_ingest.image_pixel_loader import _resize_lanczos
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageOps import exif_transpose
import albumentations as A
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator
from toolkit.prompt_utils import PromptEmbeds
from torchvision.transforms import functional as TF

from toolkit.train_tools import get_torch_dtype


# -- phase-1 control reference latent sidecars --------------------------------
# Models with supports_control_latent_cache encode their control/reference
# images once in phase 1 (sized for the training item's bucket) instead of
# per training step. One sidecar per training item next to the latent cache;
# validity is keyed on the control files' content signatures + bucket dims,
# so edited controls or a changed bucket regenerate on the next run.

def control_latent_sidecar_path(img_path: str) -> str:
    img_dir = os.path.dirname(img_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(img_dir, '_latent_cache', f'{stem}.ctrl.safetensors')


def _signature_hash(signature: str) -> int:
    import hashlib

    return int(hashlib.md5(signature.encode('utf-8')).hexdigest(), 16) % (2 ** 63)


def _read_control_sidecar_meta(sidecar_path: str) -> dict:
    from safetensors.torch import safe_open

    with safe_open(sidecar_path, framework='pt', device='cpu') as f:
        if 'meta_json' not in f.keys():
            raise KeyError('meta_json')
        import json

        return json.loads(bytes(f.get_tensor('meta_json').tolist()).decode('utf-8'))

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.stable_diffusion_model import StableDiffusion

accelerator = get_accelerator()

# def get_associated_caption_from_img_path(img_path):
# https://demo.albumentations.ai/
class Augments:
    def __init__(self, **kwargs):
        self.method_name = kwargs.get('method', None)
        self.params = kwargs.get('params', {})

        # convert kwargs enums for cv2
        for key, value in self.params.items():
            if isinstance(value, str):
                # split the string
                split_string = value.split('.')
                if len(split_string) == 2 and split_string[0] == 'cv2':
                    if hasattr(cv2, split_string[1]):
                        self.params[key] = getattr(cv2, split_string[1].upper())
                    else:
                        raise ValueError(f"invalid cv2 enum: {split_string[1]}")


transforms_dict = {
    'ColorJitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    'RandomEqualize': transforms.RandomEqualize(p=0.2),
}

img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']
video_ext_list = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv']


def standardize_images(images):
    """
    Standardize the given batch of images using the specified mean and std.
    Expects values of 0 - 1

    Args:
    images (torch.Tensor): A batch of images in the shape of (N, C, H, W),
                           where N is the number of images, C is the number of channels,
                           H is the height, and W is the width.

    Returns:
    torch.Tensor: Standardized images.
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    # Define the normalization transform
    normalize = transforms.Normalize(mean=mean, std=std)

    # Apply normalization to each image in the batch
    standardized_images = torch.stack([normalize(img) for img in images])

    return standardized_images

def clean_caption(caption):
    # this doesnt make any sense anymore in a world that is not based on comma seperated tokens
    # # remove any newlines
    # caption = caption.replace('\n', ', ')
    # # remove new lines for all operating systems
    # caption = caption.replace('\r', ', ')
    # caption_split = caption.split(',')
    # # remove empty strings
    # caption_split = [p.strip() for p in caption_split if p.strip()]
    # # join back together
    # caption = ', '.join(caption_split)
    return caption

def waveform_to_stereo(waveform):
    c = waveform.shape[0]
    if c == 2:
        return waveform
    if c == 1:
        return waveform.expand(2, -1)
    if c == 6:  # 5.1: FL, FR, FC, LFE, BL, BR
        fl, fr, fc, _, bl, br = waveform
        k = 0.7071
        return torch.stack([fl + k * fc + k * bl, fr + k * fc + k * br])
    if c == 8:  # 7.1: FL, FR, FC, LFE, BL, BR, SL, SR
        fl, fr, fc, _, bl, br, sl, sr = waveform
        k = 0.7071
        return torch.stack([fl + k * fc + k * (bl + sl), fr + k * fc + k * (br + sr)])
    return waveform.mean(0, keepdim=True).expand(2, -1)


class CaptionMixin:
    def get_caption_item(self: 'AiToolkitDataset', index):
        if not hasattr(self, 'caption_type'):
            raise Exception('caption_type not found on class instance')
        if not hasattr(self, 'file_list'):
            raise Exception('file_list not found on class instance')
        img_path_or_tuple = self.file_list[index]
        ext = self.dataset_config.caption_ext
        if isinstance(img_path_or_tuple, tuple):
            img_path = img_path_or_tuple[0] if isinstance(img_path_or_tuple[0], str) else img_path_or_tuple[0].path
            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = None
            prompt_path = path_no_ext + ext
        else:
            img_path = img_path_or_tuple if isinstance(img_path_or_tuple, str) else img_path_or_tuple.path
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = path_no_ext + ext
                
        # allow folders to have a default prompt
        default_prompt_path = os.path.join(os.path.dirname(img_path), 'default.txt')
        default_prompt_path_with_ext = os.path.join(os.path.dirname(img_path), 'default' + ext)

        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path_with_ext):
            with open(default_prompt_path_with_ext, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path):
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        else:
            prompt = ''
            # get default_prompt if it exists on the class instance
            if hasattr(self, 'default_prompt'):
                prompt = self.default_prompt
            if hasattr(self, 'default_caption'):
                prompt = self.default_caption

        # handle replacements
        replacement_list = self.dataset_config.replacements if isinstance(self.dataset_config.replacements, list) else []
        for replacement in replacement_list:
            from_string, to_string = replacement.split('|')
            prompt = prompt.replace(from_string, to_string)

        return prompt


if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO


class Bucket:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.file_list_idx: List[int] = []


class BucketsMixin:
    def __init__(self):
        self.buckets: Dict[str, Bucket] = {}
        # each batch is a list of (file_idx, flip_x, flip_y) slots
        self.batch_indices: List[List[tuple]] = []

    def build_batch_indices(self: 'AiToolkitDataset'):
        # bucket.file_list_idx holds UNIQUE file indices. Repeats and flip
        # variants are expanded into (file_idx, flip_x, flip_y) slots here --
        # the expansion the old code performed by copying DTOs. Slots are
        # shuffled after expansion so the variants of one file scatter across
        # batches the way the old expanded-then-shuffled lists did.
        repeats = max(1, int(self.dataset_config.num_repeats))
        variants = self._slot_variants()
        self.batch_indices = []
        for key, bucket in self.buckets.items():
            slots = []
            for file_idx in bucket.file_list_idx:
                for _ in range(repeats):
                    slots.extend((file_idx, fx, fy) for fx, fy in variants)
            random.shuffle(slots)
            for start_idx in range(0, len(slots), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(slots))
                batch = slots[start_idx:end_idx]
                # if the bucket has fewer items left than the requested batch size,
                # duplicate items from this batch to pad it up to batch_size
                if len(batch) < self.batch_size and len(batch) > 0:
                    pad = [batch[i % len(batch)] for i in range(self.batch_size - len(batch))]
                    batch = batch + pad
                self.batch_indices.append(batch)

    def shuffle_buckets(self: 'AiToolkitDataset'):
        for key, bucket in self.buckets.items():
            random.shuffle(bucket.file_list_idx)

    def setup_buckets(self: 'AiToolkitDataset', quiet=False):
        if not hasattr(self, 'file_list'):
            raise Exception(f'file_list not found on class instance {self.__class__.__name__}')
        if not hasattr(self, 'dataset_config'):
            raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')

        if self.epoch_num > 0:
            # no need to rebuild buckets for now
            # todo handle random cropping for buckets
            return
        self.buckets = {}  # clear it

        config: 'DatasetConfig' = self.dataset_config
        if getattr(config, 'ragged_batches', False):
            # variable-shape batches: one group, a per-image shape, no crop.
            # Everything below is the shipped fixed-shape bucket path.
            self._setup_ragged_groups(config)
            self.shuffle_buckets()
            self.build_batch_indices()
            return

        resolution = config.resolution
        bucket_tolerance = config.bucket_tolerance
        file_list: List['FileItemDTO'] = self.file_list

        # for file_item in enumerate(file_list):
        for idx, file_item in enumerate(file_list):
            file_item: 'FileItemDTO' = file_item
            if file_item.is_audio_model:
                bucket_key = f"{file_item.width}ms"
                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = Bucket(file_item.width, 1)
                self.buckets[bucket_key].file_list_idx.append(idx)
                continue
            width = int(file_item.width * file_item.dataset_config.scale)
            height = int(file_item.height * file_item.dataset_config.scale)

            if self.dataset_config.square_crop:
                # we scale first so smallest size matches resolution
                scale_factor_x = resolution / width
                scale_factor_y = resolution / height
                scale_factor = max(scale_factor_x, scale_factor_y)
                file_item.scale_to_width = math.ceil(width * scale_factor)
                file_item.scale_to_height = math.ceil(height * scale_factor)
                file_item.crop_width = resolution
                file_item.crop_height = resolution
                if width > height:
                    file_item.crop_x = int(file_item.scale_to_width / 2 - resolution / 2)
                    file_item.crop_y = 0
                else:
                    file_item.crop_x = 0
                    file_item.crop_y = int(file_item.scale_to_height / 2 - resolution / 2)
            else:
                bucket_resolution = get_bucket_for_image_size(
                    width, height,
                    resolution=resolution,
                    divisibility=bucket_tolerance
                )

                # Calculate scale factors for width and height
                width_scale_factor = bucket_resolution["width"] / width
                height_scale_factor = bucket_resolution["height"] / height

                # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
                max_scale_factor = max(width_scale_factor, height_scale_factor)

                # round up
                file_item.scale_to_width = int(math.ceil(width * max_scale_factor))
                file_item.scale_to_height = int(math.ceil(height * max_scale_factor))

                file_item.crop_height = bucket_resolution["height"]
                file_item.crop_width = bucket_resolution["width"]

                new_width = bucket_resolution["width"]
                new_height = bucket_resolution["height"]

                if self.dataset_config.random_crop:
                    # random crop
                    crop_x = random.randint(0, file_item.scale_to_width - new_width)
                    crop_y = random.randint(0, file_item.scale_to_height - new_height)
                    file_item.crop_x = crop_x
                    file_item.crop_y = crop_y
                else:
                    # do central crop
                    file_item.crop_x = int((file_item.scale_to_width - new_width) / 2)
                    file_item.crop_y = int((file_item.scale_to_height - new_height) / 2)

                if file_item.crop_y < 0 or file_item.crop_x < 0:
                    print_acc('debug')

            # check if bucket exists, if not, create it
            bucket_key = f'{file_item.crop_width}x{file_item.crop_height}'
            if self.is_video:
                # images (1 frame) and videos must not mix in a batch
                bucket_key += f'x{file_item.num_frames}f'
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(file_item.crop_width, file_item.crop_height)
            self.buckets[bucket_key].file_list_idx.append(idx)

        # print the buckets
        self.shuffle_buckets()
        self.build_batch_indices()
        if not quiet:
            print_acc(f'Bucket sizes for {self.dataset_path}:')
            for key, bucket in self.buckets.items():
                print_acc(f'{key}: {len(bucket.file_list_idx)} files')
            print_acc(f'{len(self.buckets)} buckets made')

    def _setup_ragged_groups(self: 'AiToolkitDataset', config):
        """Ragged batching (opt-in, dataset_config.ragged_batches): give each
        image its OWN shape -- native aspect ratio snapped to the model's
        divisibility grid at the area budget -- and put every usable image in
        ONE group, so a batch mixes shapes instead of carrying one shape each.

        The shipped resize+crop then becomes resize-only by construction: the
        crop rect equals the scaled size at the origin, so nothing is cropped
        and the image trains as authored. The shape chooser itself is reused
        verbatim (get_bucket_for_image_size), so the shapes this picks are the
        shapes the bucket path would have picked -- only the grouping changes.

        Areas stay within ragged_area_tolerance of the dataset's largest area
        so one batch's token total stays near the anchor and batch_size keeps
        meaning what it means. Images outside the window are removed from the
        dataset (never trained, never cached) with a printed count-and-area
        summary; they are never silently trained at a different token cost.

        Anything this path cannot carry raises here. Silently training
        something other than what the config asked for is the failure mode
        worth avoiding.
        """
        if self.is_video:
            raise ValueError(
                "ragged_batches does not support video datasets (frames are "
                "batched as a rectangle by design)")
        if config.square_crop:
            raise ValueError(
                "ragged_batches and square_crop contradict each other: one "
                "crops every image to a square, the other keeps its shape")
        if config.random_crop:
            raise ValueError(
                "ragged_batches never crops, so random_crop has nothing to "
                "randomize. Remove random_crop or the ragged_batches flag")
        if config.random_scale:
            raise ValueError(
                "ragged_batches sizes each image to its own fixed shape; "
                "random_scale would vary it per load and invalidate the cached "
                "geometry. Remove random_scale")
        divisibility = config.bucket_tolerance
        if divisibility < 2:
            raise ValueError(
                f"ragged_batches needs the model's pixel divisibility, got "
                f"{divisibility}")

        chosen = []
        for idx, file_item in enumerate(self.file_list):
            if getattr(file_item, 'is_audio_model', False):
                raise ValueError(
                    "ragged_batches does not support audio datasets")
            width = int(file_item.width * config.scale)
            height = int(file_item.height * config.scale)
            bucket_resolution = get_bucket_for_image_size(
                width, height,
                resolution=config.resolution,
                divisibility=divisibility,
            )
            bw = int(bucket_resolution["width"])
            bh = int(bucket_resolution["height"])
            chosen.append((idx, bw, bh, bw * bh))
        if not chosen:
            raise ValueError("ragged_batches: dataset has no images to group")

        anchor = max(a for _, _, _, a in chosen)
        floor = anchor * (1.0 - float(config.ragged_area_tolerance))
        kept = [c for c in chosen if c[3] >= floor]
        skipped = [c for c in chosen if c[3] < floor]
        if not kept:
            raise ValueError(
                f"ragged_batches: every image falls below area {floor:.0f} "
                f"(anchor {anchor}); check ragged_area_tolerance and resolution")

        if skipped:
            # "not trained" must mean GONE from file_list, not just out of the
            # batches: the latent/TE cache passes walk file_list, and a
            # skipped file keeps its legacy native-sized geometry, which many
            # models cannot even encode (e.g. a VAE that patch-packs latents
            # 2x2 rejects a side not divisible by its grid). Removing here
            # keeps every downstream count: trained == cached == warned.
            dropped_idx = {c[0] for c in skipped}
            remap = {}
            new_file_list = []
            for i, item in enumerate(self.file_list):
                if i in dropped_idx:
                    continue
                remap[i] = len(new_file_list)
                new_file_list.append(item)
            self.file_list = new_file_list
            kept = [(remap[i], w, h, a) for (i, w, h, a) in kept]

        for idx, bw, bh, _area in kept:
            file_item = self.file_list[idx]
            # crop rect == scaled size at the origin: an identity crop
            file_item.scale_to_width = bw
            file_item.scale_to_height = bh
            file_item.crop_width = bw
            file_item.crop_height = bh
            file_item.crop_x = 0
            file_item.crop_y = 0

        # one group => build_batch_indices slices it into batches that mix
        # shapes, which is what the packed DiT consumes
        key = f'ragged_{config.resolution}'
        self.buckets[key] = Bucket(kept[0][1], kept[0][2])
        self.buckets[key].file_list_idx = [idx for idx, _, _, _ in kept]

        areas = [a for _, _, _, a in kept]
        print_acc(
            f'[ragged] {len(kept)} images in one mixed-shape group '
            f'(resolution {config.resolution}, area {min(areas)}-{max(areas)}, '
            f'anchor {anchor}, tolerance {config.ragged_area_tolerance:.2f})')
        if skipped:
            sa = [a for _, _, _, a in skipped]
            print_acc(
                f'[ragged] WARNING: {len(skipped)} images removed from the '
                f'dataset, area below {floor:.0f} (range {min(sa)}-{max(sa)} '
                f'vs anchor {anchor}). They are NOT trained and NOT cached. '
                f'Raise ragged_area_tolerance or resize them to the area '
                f'budget.')


# ---------------------------------------------------------------------------
# Caption-slot texts -- ONE home (owner contract 2026-10-27;
# docs/refactor_notes/DECISION--blank-embeds-and-dop-gating.md). The dataset
# DTO, the on-disk cache planner/reader, and SDTrainer's fixed-prompt cache
# all take these texts from HERE; nothing re-derives them from raw config.
#   BLANK / dropout : ALWAYS ''. Putting text into a dropout caption (the
#       trigger word, a class, a token join) is no longer dropout -- it lies
#       about the unconditional training class (owner: "inserting text into
#       dropout means it is no longer dropout and it is lying").
#   DOP             : NOT dropout. Pair for real captions is built by
#       load_caption (trigger-resolved + class-resolved, in place). These
#       two cover the caption-less sides: the class side of a DROPPED
#       caption, and the trainer's fixed prior fallback.
# D-OPSD keeps its own fenced teacher text (get_dopsd_dropout_caption); do
# not route it through this contract, and never route this through it.
# ---------------------------------------------------------------------------

BLANK_CAPTION_TEXT = ''


def dop_dropout_class_text(inline_pairs, trigger_word, class_name, is_reg=False):
    # Class side of a DROPPED caption on a DOP step: inline classes as a
    # serial-comma list ('a, b and c' -- grammatical at any tag count),
    # config-trigger datasets as the bare preservation class, everything
    # else '' (which aliases onto the shared blank).
    if inline_pairs is not None and not is_reg:
        classes = list(dict.fromkeys([c for t, c in inline_pairs]))
        if len(classes) > 2:
            return ', '.join(classes[:-1]) + ' and ' + classes[-1]
        return ' and '.join(classes)
    if trigger_word is not None and not is_reg:
        return class_name
    return ''


def dop_prior_fallback_text(class_name):
    # The trainer's fixed-cache 'dop_class' prior: conditions on the bare
    # preservation class when no per-item class-side embed exists (live
    # mode). Fixed prompt, not a dropout slot.
    return class_name


def dop_class_side_prompt(prompt, inline_pairs, trigger_word, class_name):
    # Class-side conditioning for a DOP regularisation step, derived from the
    # ALREADY-BUILT LIVE prompt (cached mode pre-encodes the pair and never
    # comes here). load_caption's caption_dop is not reused: the live prompt
    # carries per-step token ops (token dropout, shuffles), so the swap rides
    # the live text (owner contract 2026-10-27). Inline [trigger:X class:Y]
    # tokens were resolved to their triggers in the trained prompt -- swap
    # those back to classes; config-trigger datasets then swap the config
    # trigger for the preservation class.
    if inline_pairs is not None:
        for pair_trigger, pair_class in inline_pairs:
            prompt = prompt.replace(pair_trigger, pair_class)
    if trigger_word is not None:
        prompt = prompt.replace(trigger_word, class_name)
    return prompt


# One row per text-embed variant an item can have (see
# TextEmbeddingFileItemDTOMixin.get_embed_variants -- the single table the
# phase-2 planner, the train-time reader, and the completeness gate share).
EmbedVariant = namedtuple('EmbedVariant', 'name path caption dropout_plain self_ref')


class CaptionProcessingDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
            self.raw_caption: str = None
            self.raw_caption_short: str = None
            self.caption: str = None
            self.caption_short: str = None
            # caption with the trigger word replaced by the diff output preservation class
            self.caption_dop: str = None
            # D-OPSD teacher caption (trigger word replaced by <Picture 1>/<Video 1>)
            self.caption_dopsd: str = None
            # parsed inline [trigger:X class:Y] tokens [(trigger, class), ...] and the
            # caption variant with the classes swapped in (None when no inline tokens)
            self._inline_dop_pairs = None
            self._inline_dop_caption = None

            dataset_config: DatasetConfig = kwargs.get('dataset_config', None)
            self.extra_values: List[float] = dataset_config.extra_values
            self.trigger_word = dataset_config.trigger_word

    # todo allow for loading from sd-scripts style dict
    def load_caption(self: 'FileItemDTO', caption_dict: Union[dict, None]=None):
        if self.raw_caption is not None:
            # we already loaded it
            pass
        elif caption_dict is not None and self.path in caption_dict and "caption" in caption_dict[self.path]:
            self.raw_caption = caption_dict[self.path]["caption"]
            if 'caption_short' in caption_dict[self.path]:
                self.raw_caption_short = caption_dict[self.path]["caption_short"]
                if self.dataset_config.use_short_captions:
                    self.raw_caption = caption_dict[self.path]["caption_short"]
        else:
            # see if prompt file exists
            path_no_ext = os.path.splitext(self.path)[0]
            prompt_ext = self.dataset_config.caption_ext
            prompt_path = path_no_ext + prompt_ext
            short_caption = None

            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                    short_caption = None
                    prompt = clean_caption(prompt)
                    if short_caption is not None:
                        short_caption = clean_caption(short_caption)
                    
                    if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                        prompt = self.dataset_config.default_caption
            else:
                prompt = ''
                if self.dataset_config.default_caption is not None:
                    prompt = self.dataset_config.default_caption

            if short_caption is None:
                short_caption = self.dataset_config.default_caption
            self.raw_caption = prompt
            self.raw_caption_short = short_caption

        processed_caption = self.get_caption()
        self._inline_dop_pairs = extract_inline_dop_pairs(processed_caption) or None
        self._inline_dop_caption = None
        if self._inline_dop_pairs is not None:
            # swap the inline tokens after all caption processing so both variants
            # keep identical token order. The normal caption ALWAYS takes the
            # trigger word -- this resolution is not DOP-gated: the
            # trigger-resolved caption is the one and only embed when DOP is
            # off. The class-swapped variant feeds the DOP branch below and
            # nothing else.
            # contract: docs/refactor_notes/DECISION--blank-embeds-and-dop-gating.md
            self.caption = apply_inline_dop(processed_caption, use_class=False)
            self._inline_dop_caption = apply_inline_dop(processed_caption, use_class=True)
        else:
            self.caption = processed_caption
        if self.raw_caption_short is not None:
            self.caption_short = self.get_caption(short_caption=True)
            if self._inline_dop_pairs is not None:
                self.caption_short = apply_inline_dop(self.caption_short, use_class=False)
        if self.dataset_config.diff_output_preservation:
            # DOP GATE: the class-side caption exists ONLY here. Inline tokens
            # are inert and no class embed is produced anywhere unless DOP is
            # on (contract: docs/refactor_notes/DECISION--blank-embeds-and-dop-gating.md).
            # Replace this dataset's trigger word with the preservation class.
            # do it on the final caption so token order matches the normal caption
            # (if the caption has inline tokens, start from the class-swapped variant)
            self.caption_dop = self._inline_dop_caption if self._inline_dop_caption is not None else self.caption
            if self.trigger_word is not None:
                self.caption_dop = self.caption_dop.replace(
                    self.trigger_word, self.dataset_config.diff_output_preservation_class
                )
        if getattr(self, 'dopsd_self_ref', False):
            # trigger word -> the self-reference token, or the token prepended
            # when there is no trigger word
            if self.trigger_word is not None:
                self.caption_dopsd = self.caption.replace(
                    self.trigger_word, self.get_dopsd_ref_token()
                )
            else:
                self.caption_dopsd = f"{self.get_dopsd_ref_token()} {self.caption}".strip()

    def get_dopsd_ref_token(self: 'FileItemDTO') -> str:
        # the item is always the only reference in D-OPSD mode
        return "<Video 1>" if self.is_video else "<Picture 1>"

    def _drop_caption_tokens(self: 'FileItemDTO', token_list):
        # token-level dropout: the first keep_tokens tags always survive; the
        # rest drop independently at token_dropout_rate (rate >= 1.0 drops
        # every non-kept token).
        keep_tokens: int = self.dataset_config.keep_tokens
        rate = self.dataset_config.token_dropout_rate
        kept = []
        for idx, token in enumerate(token_list):
            if idx < keep_tokens:
                kept.append(token)
            elif rate >= 1.0:
                # drop the token
                pass
            else:
                # get a random float form 0 to 1
                rand = random.random()
                if rand > rate:
                    # keep the token
                    kept.append(token)
        return kept

    def _apply_tag_token_ops(self: 'FileItemDTO', raw_caption, do_token_dropout):
        # tag-style token ops (dropout / shuffle) split on commas; anything
        # else keeps the text exactly as written (splitting and re-joining
        # with ", " doubled the space after every comma in prose captions)
        if not (do_token_dropout or self.dataset_config.shuffle_tokens):
            return raw_caption
        token_list = [t.strip() for t in raw_caption.split(',')]
        if do_token_dropout:
            token_list = self._drop_caption_tokens(token_list)
        if self.dataset_config.shuffle_tokens:
            random.shuffle(token_list)
        return ', '.join(token_list)

    def _append_random_triggers(self: 'FileItemDTO', caption):
        # random_triggers datasets append a random sample of their trigger
        # list as trailing tags (0..max, uniform)
        num_triggers = self.dataset_config.random_triggers_max
        if num_triggers > 1:
            num_triggers = random.randint(0, num_triggers)
        if num_triggers <= 0:
            return caption
        triggers = random.sample(self.dataset_config.random_triggers, num_triggers)
        return caption + ', ' + ', '.join(triggers)

    def get_caption(
            self: 'FileItemDTO',
            trigger=None,
            to_replace_list=None,
            add_if_not_present=False,
            short_caption=False
    ):
        if trigger is None and self.trigger_word is not None:
            trigger = self.trigger_word
        
        if trigger is not None and not self.is_reg:
            # add if not present if not regularization
            add_if_not_present = True
            
        if short_caption:
            raw_caption = self.raw_caption_short
        else:
            raw_caption = self.raw_caption
        if raw_caption is None:
            raw_caption = ''
        # handle dropout (live mode -- cached mode swaps embeds in
        # load_prompt_embedding). Either way, dropout means DROPPED: the text
        # is '' and nothing is injected into it (owner contract
        # 2026-10-27). Record the decision so this batch's DOP regularisation
        # side can condition on the class instead -- DOP is not dropout.
        if not short_caption:
            self._caption_was_dropped = False
        if self.dataset_config.caption_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
            # get a random float form 0 to 1
            rand = random.random()
            if rand < self.dataset_config.caption_dropout_rate:
                # drop the caption
                self._caption_was_dropped = True
                return ''

        do_token_dropout = (
            self.dataset_config.token_dropout_rate > 0
            and not short_caption
            and not self.dataset_config.cache_text_embeddings
        )
        caption = self._apply_tag_token_ops(raw_caption, do_token_dropout)

        if add_if_not_present and len(extract_inline_dop_pairs(caption)) > 0:
            # the inline [trigger:X class:Y] tokens bring their own trigger word.
            # do not auto-prepend the config trigger on top of them
            add_if_not_present = False
        caption = inject_trigger_into_prompt(caption, trigger, to_replace_list, add_if_not_present)

        if self.dataset_config.random_triggers:
            caption = self._append_random_triggers(caption)

        if self.dataset_config.shuffle_tokens:
            # shuffle again (after injection so the trigger shuffles with the
            # rest of the tags)
            token_list = [t.strip() for t in caption.split(',')]
            random.shuffle(token_list)
            caption = ', '.join(token_list)
        return caption

class AudioProcessingDTOMixin:
    def load_and_process_audio(self: 'FileItemDTO'):
        # Default to "no audio" unless we successfully extract it
        self.audio_data = None
        self.audio_tensor = None
        self.tensor = None
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(self.path)  # [channels, samples]
            waveform = waveform_to_stereo(waveform)  # Convert to stereo if not already
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            self.tensor = waveform
            self.audio_tensor = waveform
            self.audio_data = {"waveform": waveform, "sample_rate": int(self.sample_rate)}

        except Exception as e:
            # if issue with libtorchcodec "Could not load libtorchcodec"
            raise Exception(f"** WARNING ** - Error Processing audio for {self.path}. Error: {e}")
        

class ImageProcessingDTOMixin:
    def get_auto_frame_count(self: 'FileItemDTO', total_frames: int, video_fps: float) -> int:
        # frame count this video will train at with auto_frame_count. Also called at
        # FileItemDTO init so bucket keys carry the real frame count, so it must give
        # the same answer at bucketing time and at load time.
        # allow for any length video here but make sure it is temporally compressable.
        # The math lives in toolkit.video_ingest (the float-ingest port) so the
        # bucketing and the decode share one implementation.
        return compute_auto_frame_count(
            total_frames=total_frames,
            video_fps=video_fps,
            dataset_fps=self.dataset_config.fps,
            temporal_compression=self.temporal_compression,
            frame_count_snapper=getattr(self, 'frame_count_snapper', None),
            trim_auto_frame_count_tail=getattr(
                self.dataset_config, 'trim_auto_frame_count_tail', False
            ),
        )

    def load_and_process_video(
        self: 'FileItemDTO',
        transform: Union[None, transforms.Compose],
        only_load_latents=False
    ):
        
        if self.augments is not None and len(self.augments) > 0:
            raise Exception('Augments not supported for videos')
            
        if self.has_augmentations:
            raise Exception('Augmentations not supported for videos')
        
        if not self.dataset_config.buckets:
            raise Exception('Buckets required for video processing')
        
        do_audio = self.dataset_config.do_audio
        
        try:
            # Float ingest: ffprobe -> the fork's own frame-index semantics ->
            # one ffmpeg pass decoding the bucket geometry as fp32 gbrpf32le.
            # No cv2, no PIL, no uint8 anywhere between the file and the vae;
            # dataset clips are already framed (the export owns any crop),
            # so the decode only resizes to the assigned bucket -- no crop
            # filter exists on this path.
            loaded = load_dataset_float_tensors(
                self.path,
                bucket_width=self.crop_width,
                bucket_height=self.crop_height,
                dataset_fps=self.dataset_config.fps,
                num_frames=self.num_frames,
                auto_frame_count=self.dataset_config.auto_frame_count,
                shrink_video_to_frames=self.dataset_config.shrink_video_to_frames,
                trim_auto_frame_count_tail=getattr(
                    self.dataset_config, "trim_auto_frame_count_tail", False
                ),
                temporal_compression=self.temporal_compression,
                frame_count_snapper=getattr(self, "frame_count_snapper", None),
            )
            tensor = loaded.tensor
            # flips are exact in fp32 after the decode (the kernels the graph
            # uses are symmetric, so flip-after-geometry == flip-before)
            if self.flip_x:
                tensor = torch.flip(tensor, dims=[3])
            if self.flip_y:
                tensor = torch.flip(tensor, dims=[2])

            # Stack frames into tensor [frames, channels, height, width]
            self.tensor = tensor
            self.num_frames = loaded.num_frames
            video_fps = loaded.video_fps
            frames_to_extract = loaded.frames_to_extract

            # ------------------------------
            # Audio extraction + stretching
            # ------------------------------
            if do_audio:
                # Default to "no audio" unless we successfully extract it
                self.audio_data = None
                self.audio_tensor = None

                if (self.dataset_config.auto_frame_count
                        and getattr(self.dataset_config, "trim_auto_frame_count_tail", False)):
                    # the imported audio path implements the plain
                    # slice-and-stretch contract only; the trim-tail audio
                    # variant is not implemented in the float ingest
                    raise NotImplementedError(
                        f"{self.path}: trim_auto_frame_count_tail audio is not "
                        "supported by the float ingest; set "
                        "trim_auto_frame_count_tail: false in the dataset config"
                    )

                try:
                    audio = load_clip_audio(
                        self.path,
                        frames_to_extract=frames_to_extract,
                        video_fps=video_fps,
                        num_frames=self.num_frames,
                        dataset_fps=self.dataset_config.fps,
                        normalize=self.dataset_config.audio_normalize,
                        preserve_pitch=self.dataset_config.audio_preserve_pitch,
                        preserve_pitch_fn=(
                            lambda wf, sr, target: time_stretch_preserve_pitch(wf, sr, target)
                        ),
                    )

                except Exception as e:
                    # if issue with libtorchcodec "Could not load libtorchcodec"
                    raise Exception(f"** WARNING ** - Error Processing audio for {self.path}. Error: {e}")

                if audio is not None:
                    waveform = audio["waveform"]
                    self.audio_tensor = waveform
                    self.audio_data = {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}

            # Only log success in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"Successfully loaded video with {self.tensor.shape[0]} frames: {self.path}")

        except Exception as e:
            error_msg = str(e)
            print_acc(f"Error: {error_msg}")
            print_acc(f"Error loading video: {self.path}")

            # Re-raise with more detailed information
            raise Exception(f"Video loading error ({self.path}): {error_msg}") from e
        
    def load_and_process_image(
            self: 'FileItemDTO',
            transform: Union[None, transforms.Compose],
            only_load_latents=False
    ):
        # handle get_prompt_embedding
        if self.is_text_embedding_cached:
            self.load_prompt_embedding()
        # if we are caching latents, just do that
        if self.is_latent_cached:
            self.get_latent()
            # if load_image_when_caching_latents is set, we still need the raw image
            # tensor in addition to the cached latent, so fall through to load it below
            if not self.dataset_config.load_image_when_caching_latents:
                if self.has_control_image:
                    # phase-1 cached reference latents win when present; the
                    # per-fetch pixel decode is skipped entirely
                    if not self.load_control_latents():
                        self.load_control_image()
                if self.has_inpaint_image:
                    self.load_inpaint_image()
                if self.has_clip_image:
                    self.load_clip_image()
                if self.has_mask_image:
                    self.load_mask_image()
                if self.has_unconditional:
                    self.load_unconditional_image()
                return
        if getattr(self, "is_conversation", False):
            raise RuntimeError(
                f"conversation item {self.path} has no media tensor; its "
                "training sequence comes from the phase-1 artifact"
            )
        if self.is_audio_model:
            self.load_and_process_audio()
            return
        if self.is_video:
            self.load_and_process_video(transform, only_load_latents)
            return
        try:
            img = Image.open(self.path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.path}")

        if self.use_alpha_as_mask:
            # we do this to make sure it does not replace the alpha with another color
            # we want the image just without the alpha channel
            np_img = np.array(img)
            # strip off alpha
            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        # load_rgba keeps a 4th channel for models with an RGBA VAE. A mask
        # source of 'alpha' consumes the alpha itself, so the two are mutually
        # exclusive.
        if self.load_rgba and not self.use_alpha_as_mask:
            # sources without alpha get a fully opaque one
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            # scale and crop based on file item -- the resize is the vendored
            # lanczos (with its same-size no-op guard), replacing BICUBIC;
            # the crop coords stay the fork's scaled-frame bookkeeping
            img = _resize_lanczos(img, self.scale_to_width, self.scale_to_height)
            # crop to x_crop, y_crop, x_crop + crop_width, y_crop + crop_height
            if img.width < self.crop_x + self.crop_width or img.height < self.crop_y + self.crop_height:
                # todo look into this. This still happens sometimes
                print_acc('size mismatch')
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))

            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
        else:
            # Downscale the source image first
            # TODO this is nto right
            img = img.resize(
                (int(img.size[0] * self.dataset_config.scale), int(img.size[1] * self.dataset_config.scale)),
                Image.BICUBIC)
            min_img_size = min(img.size)
            if self.dataset_config.random_crop:
                if self.dataset_config.random_scale and min_img_size > self.dataset_config.resolution:
                    if min_img_size < self.dataset_config.resolution:
                        print_acc(
                            f"Unexpected values: min_img_size={min_img_size}, self.resolution={self.dataset_config.resolution}, image file={self.path}")
                        scale_size = self.dataset_config.resolution
                    else:
                        scale_size = random.randint(self.dataset_config.resolution, int(min_img_size))
                    scaler = scale_size / min_img_size
                    scale_width = int((img.width + 5) * scaler)
                    scale_height = int((img.height + 5) * scaler)
                    img = img.resize((scale_width, scale_height), Image.BICUBIC)
                img = transforms.RandomCrop(self.dataset_config.resolution)(img)
            else:
                img = transforms.CenterCrop(min_img_size)(img)
                img = img.resize((self.dataset_config.resolution, self.dataset_config.resolution), Image.BICUBIC)

        if self.augments is not None and len(self.augments) > 0:
            # do augmentations
            for augment in self.augments:
                if augment in transforms_dict:
                    img = transforms_dict[augment](img)

        if self.has_augmentations:
            # augmentations handles transforms
            img = self.augment_image(img, transform=transform)
        elif transform:
            img = transform(img)

        self.tensor = img
        if not only_load_latents:
            if self.has_control_image:
                if not self.load_control_latents():
                    self.load_control_image()
            if self.has_inpaint_image:
                self.load_inpaint_image()
            if self.has_clip_image:
                self.load_clip_image()
            if self.has_mask_image:
                self.load_mask_image()
            if self.has_unconditional:
                self.load_unconditional_image()


class InpaintControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_inpaint_image = False
        self.inpaint_path: Union[str, None] = None
        self.inpaint_tensor: Union[torch.Tensor, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.inpaint_path is not None:
            # find the control image path
            inpaint_path = dataset_config.inpaint_path
            # we are using control images
            img_path = kwargs.get('path', None)
            img_inpaint_ext_list = ['.png', '.webp']
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]

            for ext in img_inpaint_ext_list:
                p = os.path.join(inpaint_path, file_name_no_ext + ext)
                if os.path.exists(p):
                    self.inpaint_path = p
                    self.has_inpaint_image = True
                    break
                
    def load_inpaint_image(self: 'FileItemDTO'):
        try:
            # image must have alpha channel for inpaint
            img = Image.open(self.inpaint_path)
            # make sure has aplha
            if img.mode != 'RGBA':
                return
            img = exif_transpose(img)
        
            w, h = img.size
            if w > h and self.scale_to_width < self.scale_to_height:
                # throw error, they should match
                raise ValueError(
                    f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            elif h > w and self.scale_to_height < self.scale_to_width:
                # throw error, they should match
                raise ValueError(
                    f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

            if self.flip_x:
                # do a flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.flip_y:
                # do a flip
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            if self.dataset_config.buckets:
                # scale and crop based on file item
                img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
                # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
                # crop
                img = img.crop((
                    self.crop_x,
                    self.crop_y,
                    self.crop_x + self.crop_width,
                    self.crop_y + self.crop_height
                ))
            else:
                raise Exception("Inpaint images not supported for non-bucket datasets")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if self.aug_replay_spatial_transforms:
                tensor = self.augment_spatial_control(img, transform=transform)
            else:
                tensor = transform(img)
            
            # is 0 to 1 with alpha
            self.inpaint_tensor = tensor
        
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.inpaint_path}")

    
    def cleanup_inpaint(self: 'FileItemDTO'):
        self.inpaint_tensor = None
                

class ControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_control_image = False
        self.control_path: Union[str, List[str], None] = None
        self.control_tensor: Union[torch.Tensor, None] = None
        self.control_tensor_list: Union[List[torch.Tensor], None] = None
        # phase-1 cached clean reference latents (per ref: (C, T, h, w));
        # set by load_control_latents when the sidecar is present and valid
        self.control_latent_list: Union[List[torch.Tensor], None] = None
        # file identity per control image (see load_control_image). Models
        # that VAE-encode controls may use it to cache the latents.
        self.control_cache_keys: Union[List, None] = None
        sd = kwargs.get('sd', None)
        self.use_raw_control_images = sd is not None and sd.use_raw_control_images
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.full_size_control_images = False
        if dataset_config.control_path is not None:
            # find the control image path
            control_path_list = dataset_config.control_path
            if not isinstance(control_path_list, list):
                control_path_list = [control_path_list]
            self.full_size_control_images = dataset_config.full_size_control_images
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            
            found_control_images = []
            found_control_videos = []
            allow_video_controls = sd is not None and getattr(
                sd, 'supports_video_control_images', False)
            for control_path in control_path_list:
                for ext in img_ext_list:
                    if os.path.exists(os.path.join(control_path, file_name_no_ext + ext)):
                        found_control_images.append(os.path.join(control_path, file_name_no_ext + ext))
                        self.has_control_image = True
                        break
                else:
                    if allow_video_controls:
                        for ext in video_ext_list:
                            if os.path.exists(os.path.join(control_path, file_name_no_ext + ext)):
                                found_control_videos.append(os.path.join(control_path, file_name_no_ext + ext))
                                self.has_control_image = True
                                break
            # control VIDEO paths ride on the item; the model encodes and
            # disk-caches them on first use (see minimax_h3 ref2va)
            self.control_video_paths = found_control_videos or None
            self.control_path = found_control_images
            if len(self.control_path) == 0:
                self.control_path = None
            elif len(self.control_path) == 1:
                # only do one
                self.control_path = self.control_path[0]

        if dataset_config.control_from_same_folder:
            # assume we have them. We will pull them on load.
            self.full_size_control_images = dataset_config.full_size_control_images
            self.has_control_image = True

    def get_new_control_paths(self: 'FileItemDTO'):
        if self.dataset_config.control_from_same_folder:
            # randomly grab image paths from the same folder as if they came from control_path
            pool_folder = os.path.dirname(self.path)
            # find all images in the folder
            img_files = []
            for ext in img_ext_list:
                img_files += glob.glob(os.path.join(pool_folder, f'*{ext}'))
            # remove the current image if len is greater than 1
            if len(img_files) > 1:
                img_files.remove(self.path)
            num_controls = min(self.dataset_config.num_controls_from_same_folder, len(img_files))
            # randomly grab them
            return random.sample(img_files, num_controls)
        else:
            return self.control_path

    def load_control_image(self: 'FileItemDTO'):
        control_tensors = []
        control_path_list = self.get_new_control_paths()
        if not isinstance(control_path_list, list):
            control_path_list = [control_path_list]
        # video-only controls leave control_path as None (their latents come
        # from the ref-video cache, not this image loader)
        control_path_list = [p for p in control_path_list if p is not None]
        
        for control_path in control_path_list:
            try:
                img = Image.open(control_path)
                img = exif_transpose(img)

                if self.load_rgba:
                    # keep the alpha instead of flattening it; sources without
                    # one get a fully opaque alpha
                    img = img.convert("RGBA")
                elif img.mode in ("RGBA", "LA"):
                    # Create a background with the specified transparent color
                    transparent_color = tuple(self.dataset_config.control_transparent_color)
                    background = Image.new("RGB", img.size, transparent_color)
                    # Paste the image on top using its alpha channel as mask
                    background.paste(img, mask=img.getchannel("A"))
                    img = background
                else:
                    # Already no alpha channel
                    img = img.convert("RGB")
            except Exception as e:
                print_acc(f"Error: {e}")
                print_acc(f"Error loading image: {control_path}")
            
            if not self.full_size_control_images:
                # we just scale them to 512x512:
                w, h = img.size
                img = img.resize((512, 512), Image.BICUBIC)

            elif not self.use_raw_control_images:
                w, h = img.size
                if self.flip_x:
                    # do a flip
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.flip_y:
                    # do a flip
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)

                if self.dataset_config.buckets:
                    # scale and crop based on file item
                    img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
                    # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
                    # crop
                    img = img.crop((
                        self.crop_x,
                        self.crop_y,
                        self.crop_x + self.crop_width,
                        self.crop_y + self.crop_height
                    ))
                else:
                    raise Exception("Control images not supported for non-bucket datasets")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if self.aug_replay_spatial_transforms:
                tensor = self.augment_spatial_control(img, transform=transform)
            else:
                tensor = transform(img)
            control_tensors.append(tensor)
            
        # same-folder pairing picks random files per epoch and non-raw controls
        # are flipped/bucket-cropped with the target, so only raw disk controls
        # are a deterministic function of their file
        self.control_cache_keys = None
        if (
            len(control_tensors) == len(control_path_list)
            and self.use_raw_control_images
            and self.aug_replay_spatial_transforms is None
            and not self.dataset_config.control_from_same_folder
        ):
            keys = [
                (p, get_quick_signature_string(p)) for p in control_path_list
            ]
            if all(sig is not None for _, sig in keys):
                self.control_cache_keys = keys

        if len(control_tensors) == 0:
            self.control_tensor = None
        elif len(control_tensors) == 1:
            self.control_tensor = control_tensors[0]
        elif self.use_raw_control_images:
            # just send the list of tensors as their shapes wont match
            self.control_tensor_list = control_tensors
        else:
            self.control_tensor = torch.stack(control_tensors, dim=0)

    def cleanup_control(self: 'FileItemDTO'):
        self.control_tensor = None
        self.control_tensor_list = None
        self.control_latent_list = None
        self.control_cache_keys = None

    def control_encode_live(self: 'FileItemDTO') -> bool:
        """True when this item's controls are encoded at train time no matter
        what is cached: stochastic selection or augmentation replay makes the
        control latent a per-step function."""
        if self.control_path is None:
            return False
        if not self.use_raw_control_images:
            return True
        if self.aug_replay_spatial_transforms is not None:
            return True
        if self.dataset_config is not None and self.dataset_config.control_from_same_folder:
            return True
        return False

    def _control_latent_sidecar_path(self: 'FileItemDTO') -> str:
        return control_latent_sidecar_path(self.path)

    def control_latents_pending(self: 'FileItemDTO') -> bool:
        """True when this item's controls are cacheable (raw disk controls,
        no aug replay, no same-folder pairing) and no valid sidecar exists."""
        if self.control_path is None or not self.use_raw_control_images:
            return False
        if self.aug_replay_spatial_transforms is not None:
            return False
        if self.dataset_config is not None and self.dataset_config.control_from_same_folder:
            return False
        return not self._control_latent_sidecar_valid()

    def _control_latent_sidecar_valid(self: 'FileItemDTO') -> bool:
        """True when the sidecar exists and matches the item's current control
        files (content signatures), count and bucket sizing."""
        paths = self.control_path if isinstance(self.control_path, list) else [self.control_path]
        paths = [p for p in paths if p is not None]
        if not paths:
            return False
        sidecar = self._control_latent_sidecar_path()
        if not os.path.exists(sidecar):
            return False
        try:
            meta = _read_control_sidecar_meta(sidecar)
        except Exception:
            return False
        if meta.get('target_h') != self.crop_height or meta.get('target_w') != self.crop_width:
            return False
        stored_sigs = meta.get('sigs', [])
        if meta.get('count') != len(paths) or len(stored_sigs) != len(paths):
            return False
        # a sidecar is only valid for the latent space that encoded it: a
        # second model on the same dataset must not reuse another vae's
        # reference latents. Same identity the target latent cache keys on;
        # carried by the item (sd is not available at fetch time).
        if meta.get('vae_id') != self.vae_id:
            return False
        # with cache_key_on_content off, judge by path/count/bucket only: the
        # dataset's media may be stub-swapped after caching
        if self.dataset_config is not None and not self.dataset_config.cache_key_on_content:
            return True
        for path, stored_sig in zip(paths, stored_sigs):
            sig = get_quick_signature_string(path)
            if sig is None or _signature_hash(sig) != stored_sig:
                return False
        return True

    def load_control_latents(self: 'FileItemDTO') -> bool:
        """Load phase-1 cached control reference latents when the sidecar is
        valid. Returns True when latents were loaded (pixel controls are then
        skipped entirely); False leaves control loading to the pixel path."""
        if not self._control_latent_sidecar_valid():
            return False
        from safetensors.torch import load_file

        sidecar = self._control_latent_sidecar_path()
        state_dict = load_file(sidecar, device='cpu')
        count = _read_control_sidecar_meta(sidecar)['count']
        latents = []
        for i in range(count):
            latents.append(state_dict[f'ctrl_latent_{i}'])
        self.control_latent_list = latents
        return True


class ClipImageFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_clip_image = False
        self.clip_image_path: Union[str, None] = None
        self.clip_image_tensor: Union[torch.Tensor, None] = None
        self.clip_image_embeds: Union[dict, None] = None
        self.clip_image_embeds_unconditional: Union[dict, None] = None
        self.has_clip_augmentations = False
        self.clip_image_aug_transform: Union[None, A.Compose] = None
        self.clip_image_processor: Union[None, CLIPImageProcessor] = None
        self.clip_image_encoder_path: Union[str, None] = None
        self.is_caching_clip_vision_to_disk = False
        self.is_vision_clip_cached = False
        self.clip_vision_is_quad = False
        self.clip_vision_load_device = 'cpu'
        self.clip_vision_unconditional_paths: Union[List[str], None] = None
        self._clip_vision_embeddings_path: Union[str, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.clip_image_path is not None or dataset_config.clip_image_from_same_folder:
            # copy the clip image processor so the dataloader can do it
            sd = kwargs.get('sd', None)
            if hasattr(sd.adapter, 'clip_image_processor'):
                self.clip_image_processor = sd.adapter.clip_image_processor
        if dataset_config.clip_image_path is not None:
            # find the control image path
            clip_image_path = dataset_config.clip_image_path
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(clip_image_path, file_name_no_ext + ext)):
                    self.clip_image_path = os.path.join(clip_image_path, file_name_no_ext + ext)
                    self.has_clip_image = True
                    break
            self.build_clip_imag_augmentation_transform()
            
        if dataset_config.clip_image_from_same_folder:
            # assume we have one. We will pull it on load.
            self.has_clip_image = True
            self.build_clip_imag_augmentation_transform()

    def build_clip_imag_augmentation_transform(self: 'FileItemDTO'):
        if self.dataset_config.clip_image_augmentations is not None and len(self.dataset_config.clip_image_augmentations) > 0:
            self.has_clip_augmentations = True
            augmentations = [Augments(**aug) for aug in self.dataset_config.clip_image_augmentations]

            if self.dataset_config.clip_image_shuffle_augmentations:
                random.shuffle(augmentations)

            augmentation_list = []
            for aug in augmentations:
                # make sure method name is valid
                assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
                # get the method
                method = getattr(A, aug.method_name)
                # add the method to the list
                augmentation_list.append(method(**aug.params))

            self.clip_image_aug_transform = A.Compose(augmentation_list)

    def augment_clip_image(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose], ):
        if self.dataset_config.clip_image_shuffle_augmentations:
            self.build_clip_imag_augmentation_transform()

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        if self.clip_vision_is_quad:
            # image is in a 2x2 gris. split, run augs, and recombine
            # split
            img1, img2 = np.hsplit(open_cv_image, 2)
            img1_1, img1_2 = np.vsplit(img1, 2)
            img2_1, img2_2 = np.vsplit(img2, 2)
            # apply augmentations
            img1_1 = self.clip_image_aug_transform(image=img1_1)["image"]
            img1_2 = self.clip_image_aug_transform(image=img1_2)["image"]
            img2_1 = self.clip_image_aug_transform(image=img2_1)["image"]
            img2_2 = self.clip_image_aug_transform(image=img2_2)["image"]
            # recombine
            augmented = np.vstack((np.hstack((img1_1, img1_2)), np.hstack((img2_1, img2_2))))

        else:
            # apply augmentations
            augmented = self.clip_image_aug_transform(image=open_cv_image)["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)

        return augmented_tensor

    def get_clip_vision_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("image_encoder_path", self.clip_image_encoder_path),
            ("filename", os.path.basename(self.clip_image_path)),
            ("is_quad", self.clip_vision_is_quad)
        ])
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        return item
    def get_clip_vision_embeddings_path(self: 'FileItemDTO', recalculate=False):
        if self._clip_vision_embeddings_path is not None and not recalculate:
            return self._clip_vision_embeddings_path
        else:
            # we store latents in a folder in same path as image called _latent_cache
            img_dir = os.path.dirname(self.clip_image_path)
            latent_dir = os.path.join(img_dir, '_clip_vision_cache')
            hash_dict = self.get_clip_vision_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.clip_image_path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._clip_vision_embeddings_path = os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._clip_vision_embeddings_path
    
    def get_new_clip_image_path(self: 'FileItemDTO'):
        if self.dataset_config.clip_image_from_same_folder:
            # randomly grab an image path from the same folder
            pool_folder = os.path.dirname(self.path)
            # find all images in the folder
            img_files = []
            for ext in img_ext_list:
                img_files += glob.glob(os.path.join(pool_folder, f'*{ext}'))
            # remove the current image if len is greater than 1
            if len(img_files) > 1:
                img_files.remove(self.path)
            # randomly grab one
            return random.choice(img_files)
        else:
            return self.clip_image_path

    def load_clip_image(self: 'FileItemDTO'):
        is_dynamic_size_and_aspect = isinstance(self.clip_image_processor, PixtralVisionImagePreprocessorCompatible) or \
                                    isinstance(self.clip_image_processor, SiglipImageProcessor)
        if self.clip_image_processor is None:
            is_dynamic_size_and_aspect = True # serving it raw
        if self.is_vision_clip_cached:
            self.clip_image_embeds = load_file(self.get_clip_vision_embeddings_path())

            # get a random unconditional image
            if self.clip_vision_unconditional_paths is not None:
                unconditional_path = random.choice(self.clip_vision_unconditional_paths)
                self.clip_image_embeds_unconditional = load_file(unconditional_path)

            return
        clip_image_path = self.get_new_clip_image_path()
        try:
            img = Image.open(clip_image_path).convert('RGB')
            img = exif_transpose(img)
        except Exception as e:
            # make a random noise image
            img = Image.new('RGB', (self.dataset_config.resolution, self.dataset_config.resolution))
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {clip_image_path}")

        img = img.convert('RGB')

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
        if is_dynamic_size_and_aspect:
            pass  # let the image processor handle it
        elif img.width != img.height:
            min_size = min(img.width, img.height)
            if self.dataset_config.square_crop:
                # center crop to a square
                img = transforms.CenterCrop(min_size)(img)
            else:
                # image must be square. If it is not, we will resize/squish it so it is, that way we don't crop out data
                # resize to the smallest dimension
                img = img.resize((min_size, min_size), Image.BICUBIC)

        if self.has_clip_augmentations:
            self.clip_image_tensor = self.augment_clip_image(img, transform=None)
        else:
            self.clip_image_tensor = transforms.ToTensor()(img)

        # random crop
        # if self.dataset_config.clip_image_random_crop:
        #     # crop up to 20% on all sides. Keep is square
        #     crop_percent = random.randint(0, 20) / 100
        #     crop_width = int(self.clip_image_tensor.shape[2] * crop_percent)
        #     crop_height = int(self.clip_image_tensor.shape[1] * crop_percent)
        #     crop_left = random.randint(0, crop_width)
        #     crop_top = random.randint(0, crop_height)
        #     crop_right = self.clip_image_tensor.shape[2] - crop_width - crop_left
        #     crop_bottom = self.clip_image_tensor.shape[1] - crop_height - crop_top
        #     if len(self.clip_image_tensor.shape) == 3:
        #         self.clip_image_tensor = self.clip_image_tensor[:, crop_top:-crop_bottom, crop_left:-crop_right]
        #     elif len(self.clip_image_tensor.shape) == 4:
        #         self.clip_image_tensor = self.clip_image_tensor[:, :, crop_top:-crop_bottom, crop_left:-crop_right]

        if self.clip_image_processor is not None:
            # run it
            tensors_0_1 = self.clip_image_tensor.to(dtype=torch.float16)
            clip_out = self.clip_image_processor(
                images=tensors_0_1,
                return_tensors="pt",
                do_resize=True,
                do_rescale=False,
            ).pixel_values
            self.clip_image_tensor = clip_out.squeeze(0).clone().detach()

    def cleanup_clip_image(self: 'FileItemDTO'):
        self.clip_image_tensor = None
        self.clip_image_embeds = None




class AugmentationFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_augmentations = False
        self.unaugmented_tensor: Union[torch.Tensor, None] = None
        # self.augmentations: Union[None, List[Augments]] = None
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.aug_transform: Union[None, A.Compose] = None
        self.aug_replay_spatial_transforms = None
        self.build_augmentation_transform()

    def build_augmentation_transform(self: 'FileItemDTO'):
        if self.dataset_config.augmentations is not None and len(self.dataset_config.augmentations) > 0:
            self.has_augmentations = True
            augmentations = [Augments(**aug) for aug in self.dataset_config.augmentations]

            if self.dataset_config.shuffle_augmentations:
                random.shuffle(augmentations)

            augmentation_list = []
            for aug in augmentations:
                # make sure method name is valid
                assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
                # get the method
                method = getattr(A, aug.method_name)
                # add the method to the list
                augmentation_list.append(method(**aug.params))

            # add additional targets so we can augment the control image
            self.aug_transform = A.ReplayCompose(augmentation_list, additional_targets={'image2': 'image'})

    def augment_image(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose], ):

        # rebuild each time if shuffle
        if self.dataset_config.shuffle_augmentations:
            self.build_augmentation_transform()

        # save the original tensor
        self.unaugmented_tensor = transforms.ToTensor()(img) if transform is None else transform(img)

        has_alpha = img.mode == 'RGBA'
        open_cv_image = np.array(img)
        # Convert RGB to BGR, leaving any alpha channel where it is
        if has_alpha:
            open_cv_image = open_cv_image[:, :, [2, 1, 0, 3]].copy()
        else:
            open_cv_image = open_cv_image[:, :, ::-1].copy()

        # apply augmentations
        transformed = self.aug_transform(image=open_cv_image)
        augmented = transformed["image"]

        # save just the spatial transforms for controls and masks
        augmented_params = transformed["replay"]
        spatial_transforms = ['Rotate', 'Flip', 'HorizontalFlip', 'VerticalFlip', 'Resize', 'Crop', 'RandomCrop',
                              'ElasticTransform', 'GridDistortion', 'OpticalDistortion']
        # only store the spatial transforms
        augmented_params['transforms'] = [t for t in augmented_params['transforms'] if t['__class_fullname__'].split('.')[-1] in spatial_transforms]

        if self.dataset_config.replay_transforms:
            self.aug_replay_spatial_transforms = augmented_params

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGRA2RGBA if has_alpha else cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)

        return augmented_tensor

    # augment control images spatially consistent with transforms done to the main image
    def augment_spatial_control(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose] ):
        if self.aug_replay_spatial_transforms is None:
            # no transforms
            return transform(img)

        # save colorspace to convert back to
        colorspace = img.mode
        has_alpha = colorspace == 'RGBA'

        # convert to rgb, keeping alpha so it rides the same spatial transform
        img = img.convert('RGBA' if has_alpha else 'RGB')

        open_cv_image = np.array(img)
        # Convert RGB to BGR, leaving any alpha channel where it is
        if has_alpha:
            open_cv_image = open_cv_image[:, :, [2, 1, 0, 3]].copy()
        else:
            open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Replay transforms
        transformed = A.ReplayCompose.replay(self.aug_replay_spatial_transforms, image=open_cv_image)
        augmented = transformed["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGRA2RGBA if has_alpha else cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        # convert back to original colorspace
        augmented = augmented.convert(colorspace)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)
        return augmented_tensor

    def cleanup_control(self: 'FileItemDTO'):
        self.unaugmented_tensor = None


class MaskFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_mask_image = False
        self.mask_path: Union[str, None] = None
        self.mask_tensor: Union[torch.Tensor, None] = None
        self.use_alpha_as_mask: bool = False
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.mask_min_value = dataset_config.mask_min_value
        if dataset_config.mask_source == 'alpha':
            self.use_alpha_as_mask = True
            self.mask_path = kwargs.get('path', None)
            self.has_mask_image = True
        elif dataset_config.mask_source is not None:
            # generated masks live under <image_dir>/_masks/<source>/ and are
            # produced by the phase-1 masking pass before training begins
            img_path = kwargs.get('path', None)
            mask_dir = os.path.join(
                os.path.dirname(img_path), '_masks', dataset_config.mask_source)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                candidate = os.path.join(mask_dir, file_name_no_ext + ext)
                if os.path.exists(candidate):
                    self.mask_path = candidate
                    self.has_mask_image = True
                    break
        elif dataset_config.mask_path is not None:
            # custom mask folder; files matched to images by filename stem
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(dataset_config.mask_path, file_name_no_ext + ext)):
                    self.mask_path = os.path.join(dataset_config.mask_path, file_name_no_ext + ext)
                    self.has_mask_image = True
                    break

    def load_mask_image(self: 'FileItemDTO'):
        try:
            img = Image.open(self.mask_path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        if self.use_alpha_as_mask:
            # pipeline expectws an rgb image so we need to put alpha in all channels
            np_img = np.array(img)
            np_img[:, :, :3] = np_img[:, :, 3:]

            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        if self.dataset_config.invert_mask:
            img = ImageOps.invert(img)
        w, h = img.size
        fix_size = False
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True

        if fix_size:
            # swap all the sizes
            self.scale_to_width, self.scale_to_height = self.scale_to_height, self.scale_to_width
            self.crop_width, self.crop_height = self.crop_height, self.crop_width
            self.crop_x, self.crop_y = self.crop_y, self.crop_x




        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # randomly apply a blur up to 0.5% of the size of the min (width, height)
        min_size = min(img.width, img.height)
        blur_radius = int(min_size * random.random() * 0.005)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # make grayscale
        img = img.convert('L')

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
            # crop
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            raise Exception("Mask images not supported for non-bucket datasets")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.aug_replay_spatial_transforms:
            self.mask_tensor = self.augment_spatial_control(img, transform=transform)
        else:
            self.mask_tensor = transform(img)
        self.mask_tensor = value_map(self.mask_tensor, 0, 1.0, self.mask_min_value, 1.0)
        # convert to grayscale

    def cleanup_mask(self: 'FileItemDTO'):
        self.mask_tensor = None


class UnconditionalFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_unconditional = False
        self.unconditional_path: Union[str, None] = None
        self.unconditional_tensor: Union[torch.Tensor, None] = None
        self.unconditional_latent: Union[torch.Tensor, None] = None
        self.unconditional_transforms = self.dataloader_transforms
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)

        if dataset_config.unconditional_path is not None:
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(dataset_config.unconditional_path, file_name_no_ext + ext)):
                    self.unconditional_path = os.path.join(dataset_config.unconditional_path, file_name_no_ext + ext)
                    self.has_unconditional = True
                    break

    def load_unconditional_image(self: 'FileItemDTO'):
        try:
            img = Image.open(self.unconditional_path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
            # crop
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            raise Exception("Unconditional images are not supported for non-bucket datasets")

        if self.aug_replay_spatial_transforms:
            self.unconditional_tensor = self.augment_spatial_control(img, transform=self.unconditional_transforms)
        else:
            self.unconditional_tensor = self.unconditional_transforms(img)

    def cleanup_unconditional(self: 'FileItemDTO'):
        self.unconditional_tensor = None
        self.unconditional_latent = None

class ArgBreakMixin:
    # just stops super calls form hitting object
    def __init__(self, *args, **kwargs):
        pass


def _dto_extras_from_state_dict(state_dict) -> dict:
    """Extra latent streams in a cache file: legacy named keys written by
    older versions plus the generic dto.<name> keys new caches write."""
    extras = {}
    if 'audio_latent' in state_dict:
        extras['audio'] = state_dict['audio_latent']
    for k, v in state_dict.items():
        if k.startswith(DISK_PREFIX):
            extras[k[len(DISK_PREFIX):]] = v
    return extras


class LatentCachingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        # a plain tensor, or a DTO carrying extra streams (audio rows, ...)
        self._encoded_latent: Union[torch.Tensor, None] = None
        self._cached_first_frame_latent: Union[torch.Tensor, None] = None
        # pixel tensor / waveform cached alongside the latents
        # (dataset_config.cache_tensors_to_disk), fp32, for models that need
        # pixels at train time (require_pixel_tensor_cache)
        self._cached_pixel_tensor: Union[torch.Tensor, None] = None
        self._cached_waveform: Union[torch.Tensor, None] = None
        self._cached_waveform_sample_rate: Union[int, None] = None
        self._latent_path: Union[str, None] = None
        self.is_latent_cached = False
        # guards the slot-layer flip in get_latent: the same per-fetch copy
        # is queried twice (load_and_process_image and DataLoaderBatchDTO),
        # and the flip must apply once
        self._latent_flip_applied = False
        self.is_caching_to_disk = False
        self.is_caching_to_memory = False
        self.latent_load_device = 'cpu'
        # todo, increment this if we change the latent format to invalidate cache
        self.latent_version = 1

    def get_latent_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("filename", os.path.basename(self.path)),
            ("scale_to_width", self.scale_to_width),
            ("scale_to_height", self.scale_to_height),
            ("crop_x", self.crop_x),
            ("crop_y", self.crop_y),
            ("crop_width", self.crop_width),
            ("crop_height", self.crop_height),
            ("vae_id", self.vae_id),
            ("latent_version", self.latent_version),
        ])
        if getattr(self, 'dataset_config', None) is not None and self.dataset_config.cache_key_on_content:
            # content signature of the source media: an image or video edited
            # in place lands on a fresh cache file instead of serving stale
            # tensors. Opt out for stub-slimmed datasets (see the config
            # comment) -- keying on a stub would re-encode it.
            sig = get_quick_signature_string(self.path)
            if sig is not None:
                item["source_signature"] = _signature_hash(sig)
        is_video = False
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        if self.is_video and self.dataset_config.auto_frame_count:
            # don't store num frames here as it is calculated dynamically
            item["auto_frame_count"] = True
            if self.dataset_config.trim_auto_frame_count_tail:
                # changes frame selection; only added when on so caches made before
                # this option existed stay valid when it is off
                item["trim_auto_frame_count_tail"] = True
            is_video = True
        elif self.is_video and self.dataset_config.num_frames > 1:
            item["num_frames"] = self.dataset_config.num_frames
            is_video = True
        if is_video and self.dataset_config.fps != 24:
            # only add fps if it deviates from the default
            item["fps"] = self.dataset_config.fps
        if is_video and self.dataset_config.do_i2v:
                item["do_i2v"] = True
        if is_video and self.dataset_config.do_audio:
            item["do_audio"] = True
            if self.dataset_config.audio_normalize:
                item["audio_normalize"] = True
            if self.dataset_config.audio_preserve_pitch:
                item["audio_preserve_pitch"] = True
        if self.is_audio_model:
            item["is_audio_model"] = True
            item["sample_rate"] = self.sample_rate
        if self.dataset_config.cache_tensors_to_disk:
            # pixel tensors / waveforms ride in the cache file; storing them
            # changes the key so caches made without them are not reused
            item["cache_tensors_to_disk"] = True
        if self.load_rgba:
            # the encoded alpha changes the latent; only added when on so caches
            # made before this existed stay valid for models that do not use it
            item["load_rgba"] = True
        return item

    def get_latent_path(self: 'FileItemDTO', recalculate=False):
        if self._latent_path is not None and not recalculate:
            return self._latent_path
        else:
            # we store latents in a folder in same path as image called _latent_cache
            img_dir = os.path.dirname(self.path)
            latent_dir = os.path.join(img_dir, '_latent_cache')
            hash_dict = self.get_latent_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._latent_path = os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._latent_path

    def cleanup_latent(self):
        if self._encoded_latent is not None:
            if not self.is_caching_to_memory:
                # we are caching on disk, don't save in memory
                self._encoded_latent = None
                self._cached_first_frame_latent = None
                self._cached_pixel_tensor = None
                self._cached_waveform = None
                self._cached_waveform_sample_rate = None
            else:
                # move it back to cpu (a DTO carries its extras along)
                self._encoded_latent = self._encoded_latent.to('cpu')
                if self._cached_first_frame_latent is not None:
                    self._cached_first_frame_latent = self._cached_first_frame_latent.to('cpu')

    def get_latent(self, device=None):
        if not self.is_latent_cached:
            return None
        if self._encoded_latent is None:
            # load it from disk
            state_dict = load_file(
                self.get_latent_path(),
                # device=device if device is not None else self.latent_load_device
                device='cpu'
            )
            self._encoded_latent = state_dict['latent']
            if self._encoded_latent.dtype == torch.uint8:
                # uint8 compact caches are gone (lossy truncation of the
                # vae's output); the remedy is clearing this dataset's
                # _latent_cache directory so the float path re-encodes
                raise ValueError(
                    f"{self.path}: latent cache file holds a uint8 latent -- "
                    "the uint8/int16 compact-cache options were removed; "
                    "delete this dataset's _latent_cache directory and "
                    "re-run the latent caching phase."
                )
            if 'first_frame_latent' in state_dict:
                self._cached_first_frame_latent = state_dict['first_frame_latent']
            extras = _dto_extras_from_state_dict(state_dict)
            if extras:
                self._encoded_latent = DTO(self._encoded_latent, **extras)
            if 'num_frames' in state_dict:
                self.num_frames = int(state_dict['num_frames'].item())
            if 'tensor' in state_dict:
                self._cached_pixel_tensor = state_dict['tensor']
                if self._cached_pixel_tensor.dtype == torch.uint8:
                    # a pre-fp32 cache file (the uint8 era); the remedy is
                    # clearing this dataset's _latent_cache directory
                    raise ValueError(
                        f"{self.path}: cached pixel tensor is uint8 -- the "
                        "uint8/int16 compact-cache era is gone; delete this "
                        "dataset's _latent_cache directory and re-run the "
                        "latent caching phase."
                    )
            if 'waveform' in state_dict:
                self._cached_waveform = state_dict['waveform']
                self._cached_waveform_sample_rate = int(
                    state_dict['waveform_sample_rate'].item()
                )
        pixel_tensor_rebuilt_from_cache = False
        if self._cached_pixel_tensor is not None and getattr(self, 'tensor', None) is None:
            # rebuild the pixel tensor as it would be if loaded without caching
            self.tensor = self._cached_pixel_tensor
            pixel_tensor_rebuilt_from_cache = True
        if self._cached_waveform is not None and self.audio_data is None:
            # rebuild the audio waveform as it would be if loaded without caching
            waveform = self._cached_waveform
            self.audio_tensor = waveform
            self.audio_data = {"waveform": waveform, "sample_rate": self._cached_waveform_sample_rate}
            if self.is_audio_model:
                # audio-only models use the waveform as the main tensor
                self.tensor = waveform
        if (
                self._latent_flip_is_view
                and not self._latent_flip_applied
                and (self.flip_x or self.flip_y)
                and not self.is_audio_model
        ):
            # Slot-layer flip variant. _get_single_item set these flags on the
            # per-fetch copy before it loaded; the latent cache now holds ONE
            # unflipped latent per file, so the flip is applied to the copy's
            # loaded tensors here. The live path flips pixels pre-encode;
            # flip-after-encode differs at border pixels (conv padding/bias),
            # the same accepted divergence class as crop edges. Everything
            # downstream reads these per-fetch copies, then batch.cleanup()
            # drops them, so the shared base DTO stays unflipped.
            self._latent_flip_applied = True
            if self.flip_x:
                self._encoded_latent = torch.flip(self._encoded_latent, [-1])
                if self._cached_first_frame_latent is not None:
                    self._cached_first_frame_latent = torch.flip(
                        self._cached_first_frame_latent, [-1]
                    )
                if pixel_tensor_rebuilt_from_cache:
                    self.tensor = torch.flip(self.tensor, [-1])
            if self.flip_y:
                self._encoded_latent = torch.flip(self._encoded_latent, [-2])
                if self._cached_first_frame_latent is not None:
                    self._cached_first_frame_latent = torch.flip(
                        self._cached_first_frame_latent, [-2]
                    )
                if pixel_tensor_rebuilt_from_cache:
                    self.tensor = torch.flip(self.tensor, [-2])
        return self._encoded_latent


class LatentCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.latent_cache = {}

    def latent_cache_complete(self: 'AiToolkitDataset'):
        # True when the latent cache is fully populated on disk, so a caching
        # pass would encode nothing. The training process uses this to skip
        # loading the vae entirely. Memory-only caches are rebuilt every run,
        # so they never count as complete here.
        if not self.is_caching_latents or not self.is_caching_latents_to_disk:
            return False
        for file_item in self.file_list:
            if not os.path.exists(file_item.get_latent_path(recalculate=True)):
                return False
        return True

    def cache_latents_all_latents(self: 'AiToolkitDataset'):
        with accelerator.main_process_first():
            print_acc(f"Caching latents for {self.dataset_path}")
            # cache all latents to disk
            to_disk = self.is_caching_latents_to_disk
            to_memory = self.is_caching_latents_to_memory

            if to_disk:
                print_acc(" - Saving latents to disk")
            if to_memory:
                print_acc(" - Keeping latents in memory")
            # encoding pulls the vae to the compute device as needed (see the
            # model's encode_images); no global device reshuffle is done here.

            # prep (video decode, frame extraction, audio load, disk reads) is done by a
            # thread pool so the next items are ready while the current one is encoding.
            # the in-flight window is bounded so decoded videos don't pile up in RAM.
            num_workers = max(1, self.dataset_config.cache_latents_num_workers)

            def _prep(prep_item: 'FileItemDTO'):
                prep_item.is_caching_to_disk = to_disk
                prep_item.is_caching_to_memory = to_memory
                prep_item.latent_load_device = self.sd.device

                prep_latent_path = prep_item.get_latent_path(recalculate=True)
                try:
                    if os.path.exists(prep_latent_path):
                        cached_state_dict = load_file(prep_latent_path, device='cpu') if to_memory else None
                        return prep_item, prep_latent_path, cached_state_dict, False
                    if prep_item.is_conversation:
                        # the sidecar is parsed at scan time and assembled at
                        # encode time (main thread); there is no media decode
                        return prep_item, prep_latent_path, None, True
                    # not saved to disk, load the image/video/audio
                    prep_item.load_and_process_image(self.transform, only_load_latents=True)
                except Exception as e:
                    kind = "conversation sidecar" if prep_item.is_conversation else "image"
                    print_acc(f"Error processing {kind}: {prep_item.path}")
                    print_acc(f"Error: {str(e)}")
                    print_acc(" - Skipping file and removing it from the dataset")
                    return prep_item, None, None, False
                return prep_item, prep_latent_path, None, True

            # use tqdm to show progress
            i = 0
            failed_items = []
            pbar = tqdm(total=len(self.file_list), desc=f'Caching latents{" to disk" if to_disk else ""}')
            executor = ThreadPoolExecutor(max_workers=num_workers)
            try:
                pending = deque()
                file_iter = iter(self.file_list)
                for queued_item in itertools.islice(file_iter, num_workers + 2):
                    pending.append(executor.submit(_prep, queued_item))
                while pending:
                    file_item, latent_path, cached_state_dict, needs_encode = pending.popleft().result()
                    # keep the window full
                    next_item = next(file_iter, None)
                    if next_item is not None:
                        pending.append(executor.submit(_prep, next_item))
                    if latent_path is None:
                        # file failed to load; drop it from the dataset and keep going
                        failed_items.append(file_item)
                        pbar.update(1)
                        continue
                    try:
                        self._cache_one_latent(file_item, latent_path, cached_state_dict, needs_encode, to_disk, to_memory)
                    except UnusableFileError as e:
                        pbar.write(f"Skipping unusable file {file_item.path}: {e}")
                        failed_items.append(file_item)
                        pbar.update(1)
                        continue
                    # models may report per-file repairs (e.g. a patched sheet); print them on their own line with the path
                    for warning in getattr(self.sd, 'pop_encode_warnings', lambda: [])():
                        pbar.write(f"{file_item.path}: {warning}")
                    file_item.is_latent_cached = True
                    i += 1
                    pbar.update(1)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
                pbar.close()

            if failed_items:
                print_acc(f"Removed {len(failed_items)} files from the dataset that failed to load or were unusable")
                self._remove_file_items(failed_items)

    def control_latents_pending(self: 'AiToolkitDataset') -> bool:
        """True when any cacheable control image of this dataset still lacks
        a valid reference-latent sidecar. The vae gate and the training-time
        vae residency decision both ask this."""
        if not getattr(self.sd, 'supports_control_latent_cache', False):
            return False
        if self.dataset_config.control_from_same_folder:
            return False
        return any(item.control_latents_pending() for item in self.file_list)

    def control_encode_live(self: 'AiToolkitDataset') -> bool:
        """True when any control of this dataset is a per-step function
        (stochastic selection or aug replay) and will be encoded at train
        time no matter what is cached."""
        if not getattr(self.sd, 'supports_control_latent_cache', False):
            return False
        return any(item.control_encode_live() for item in self.file_list)

    def cache_control_reference_latents(self: 'AiToolkitDataset'):
        """Phase-1 pass: encode every deterministic control image to a clean
        reference latent sidecar, sized for the owning item's bucket. Items
        with stochastic controls (same-folder pairing, aug replay) keep the
        per-step encode path and are skipped here."""
        if not getattr(self.sd, 'supports_control_latent_cache', False):
            return
        if self.dataset_config.control_from_same_folder:
            return
        device = self.sd.device_torch
        to_tensor = transforms.ToTensor()
        pending = [item for item in self.file_list if item.control_latents_pending()]
        if not pending:
            return
        print_acc(f"Caching control reference latents for {self.dataset_path}")
        for file_item in pending:
            paths = (
                file_item.control_path
                if isinstance(file_item.control_path, list)
                else [file_item.control_path]
            )
            paths = [p for p in paths if p is not None]
            if not paths:
                continue
            sigs = []
            skipped = False
            for p in paths:
                sig = get_quick_signature_string(p)
                if sig is None:
                    # unreadable control file: leave it to the per-step path
                    skipped = True
                    break
                sigs.append(_signature_hash(sig))
            if skipped:
                continue
            target_h, target_w = file_item.crop_height, file_item.crop_width
            state_dict = {}
            meta = {
                'count': len(paths),
                'sigs': sigs,
                'target_h': int(target_h),
                'target_w': int(target_w),
                # which latent space encoded these: a second model on the same
                # dataset must not reuse another vae's reference latents.
                # Same identity the target latent cache keys on (vae_id).
                'vae_id': getattr(self.sd, 'vae_id', None),
            }
            import json as _json

            state_dict['meta_json'] = torch.frombuffer(
                bytearray(_json.dumps(meta).encode('utf-8')), dtype=torch.uint8
            )
            for i, control_path in enumerate(paths):
                try:
                    img = Image.open(control_path)
                    img = exif_transpose(img).convert('RGB')
                except Exception as e:
                    print_acc(f"Error loading control image {control_path}: {e}")
                    skipped = True
                    break
                ref_w, ref_h = img.size
                ph, pw = self.sd.get_control_reference_size(
                    ref_w, ref_h, target_h, target_w)
                pixels = to_tensor(img).to(device, torch.float32)
                # bicubic to the sized target, [0, 1] -> [-1, 1], single frame
                pixels = torch.nn.functional.interpolate(
                    pixels[None],
                    size=(ph, pw),
                    mode="bicubic",
                    antialias=True,
                )[0].clamp(0.0, 1.0)
                frames = (pixels * 2.0 - 1.0).unsqueeze(1).unsqueeze(0)
                with torch.no_grad():
                    latent = self.sd.encode_control_reference(frames)
                # (1, C, T, h, w) -> (C, T, h, w), exactly what
                # _build_condition stacks back up per reference index
                state_dict[f'ctrl_latent_{i}'] = latent.squeeze(0).cpu()
            if skipped:
                continue
            sidecar = control_latent_sidecar_path(file_item.path)
            os.makedirs(os.path.dirname(sidecar), exist_ok=True)
            from safetensors.torch import save_file

            save_file(state_dict, sidecar, metadata={"format": "pt"})
        flush()

    def _remove_file_items(self: 'AiToolkitDataset', items_to_remove: List['FileItemDTO']):
        # buckets hold raw indices into file_list, so removal requires remapping them
        remove_ids = {id(item) for item in items_to_remove}
        old_to_new = {}
        new_file_list = []
        for old_idx, item in enumerate(self.file_list):
            if id(item) in remove_ids:
                continue
            old_to_new[old_idx] = len(new_file_list)
            new_file_list.append(item)
        self.file_list = new_file_list
        if self.dataset_config.buckets and getattr(self, 'buckets', None):
            for key in list(self.buckets.keys()):
                bucket = self.buckets[key]
                bucket.file_list_idx = [old_to_new[idx] for idx in bucket.file_list_idx if idx in old_to_new]
                if len(bucket.file_list_idx) == 0:
                    del self.buckets[key]
            self.build_batch_indices()
        else:
            # the non-bucket slot list indexes file_list; rebuild it after
            # the indices moved (ordering is irrelevant -- the dataloader
            # shuffles).
            self._build_slot_list()

    def _cache_one_latent(
            self: 'AiToolkitDataset',
            file_item: 'FileItemDTO',
            latent_path: str,
            cached_state_dict,
            needs_encode: bool,
            to_disk: bool,
            to_memory: bool,
    ):
        # check if it is saved to disk already
        if not needs_encode:
            if to_memory:
                # load it into memory
                state_dict = cached_state_dict
                cached_latent = state_dict['latent']
                if cached_latent.dtype == torch.uint8:
                    # uint8 compact caches are gone; the remedy is clearing
                    # this dataset's _latent_cache directory (see get_latent)
                    raise ValueError(
                        f"{file_item.path}: latent cache holds a uint8 latent "
                        "-- delete the dataset's _latent_cache directory and "
                        "re-run the latent caching phase."
                    )
                extras = _dto_extras_from_state_dict(state_dict)
                if extras:
                    cached_latent = DTO(cached_latent, **extras)
                file_item._encoded_latent = cached_latent.to('cpu')
                if 'first_frame_latent' in state_dict:
                    cached_first_frame = state_dict['first_frame_latent']
                    file_item._cached_first_frame_latent = cached_first_frame.to('cpu')
                if 'tensor' in state_dict:
                    file_item._cached_pixel_tensor = state_dict['tensor']
                if 'waveform' in state_dict:
                    file_item._cached_waveform = state_dict['waveform']
                    file_item._cached_waveform_sample_rate = int(
                        state_dict['waveform_sample_rate'].item()
                    )
        else:
            # not saved to disk, calculate
            # the image/video/audio was already loaded by the prep thread
            device = self.sd.device_torch
            state_dict = OrderedDict()
            first_frame_latent = None
            audio_latent = None
            frames = None
            # add batch dimension
            try:
                if file_item.is_conversation:
                    # the conversation artifact: the holder assembles the mixed
                    # sequence (ids/types/positions/grids/embeds/loss span) from
                    # the sidecar; caching is inference by definition
                    with torch.no_grad():
                        latent = self.sd.encode_conversation(file_item)
                else:
                    # pristine fp32 pixels into the encoder -- the vae decides
                    # its own compute dtype (the bundle casts at its boundary);
                    # waveforms were already fp32
                    imgs = file_item.tensor.unsqueeze(0).to(device, dtype=torch.float32)
                    # caching is inference by definition: the no_grad lives here so
                    # models with grad-capable encoders (future tower-LoRA) don't
                    # retain graphs during the pass
                    with torch.no_grad():
                        latent = self.sd.encode_images(imgs)
                # a model can return a DTO carrying extra streams alongside the latent
                latent = latent.map(lambda t: t.squeeze(0)) if isinstance(latent, DTO) else latent.squeeze(0)
                if to_disk:
                    main_latent = latent.tensor if isinstance(latent, DTO) else latent
                    state_dict['latent'] = main_latent.clone().detach().cpu()
                if self.dataset_config.cache_tensors_to_disk:
                    # the decoded pixels + waveform ride in the cache file at
                    # full fp32 (the float-ingest output) -- models that need
                    # pixels at train time read them from here. No compaction:
                    # the former uint8/int16 storage truncated the vae input.
                    if not file_item.is_audio_model and file_item.tensor is not None:
                        pixel_tensor = file_item.tensor.detach().to('cpu')
                        if to_disk:
                            state_dict['tensor'] = pixel_tensor
                        if to_memory:
                            file_item._cached_pixel_tensor = pixel_tensor
                    if file_item.audio_data is not None:
                        waveform = file_item.audio_data['waveform'].detach().to('cpu', dtype=torch.float32)
                        sample_rate = int(file_item.audio_data['sample_rate'])
                        if to_disk:
                            state_dict['waveform'] = waveform
                            state_dict['waveform_sample_rate'] = torch.tensor(sample_rate, dtype=torch.int32)
                        if to_memory:
                            file_item._cached_waveform = waveform
                            file_item._cached_waveform_sample_rate = sample_rate
                    if isinstance(latent, DTO):
                        for k, v in latent.extras.items():
                            if torch.is_tensor(v):
                                state_dict[f'{DISK_PREFIX}{k}'] = v.clone().detach().cpu()
            except UnusableFileError:
                raise
            except Exception as e:
                kind = "conversation sidecar" if file_item.is_conversation else "image"
                print_acc(f"Error processing {kind}: {file_item.path}")
                print_acc(f"Error: {str(e)}")
                raise e
            # do first frame
            is_video = file_item.is_video
            if is_video and self.dataset_config.do_i2v:
                frames = file_item.tensor.unsqueeze(0).to(device, dtype=torch.float32)
                if len(frames.shape) == 4:
                    first_frames = frames
                elif len(frames.shape) == 5:
                    first_frames = frames[:, 0]
                else:
                    raise ValueError(f"Unknown frame shape {frames.shape}")
                with torch.no_grad():
                    first_frame_latent = self.sd.encode_images(first_frames).squeeze(0)
                if to_disk:
                    state_dict['first_frame_latent'] = first_frame_latent.clone().detach().cpu()

            # audio (video+audio models only — audio-only models already encoded above via encode_images)
            if not file_item.is_audio_model and file_item.audio_data is not None:
                audio_latent = self.sd.encode_audio([file_item.audio_data]).squeeze(0)
                if to_disk:
                    state_dict['audio_latent'] = audio_latent.clone().detach().cpu()

            if is_video:
                state_dict['num_frames'] = torch.tensor(file_item.num_frames, dtype=torch.int32)

            # save_latent
            if to_disk:
                # metadata
                meta = get_meta_for_safetensors(file_item.get_latent_info_dict())
                os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                save_file(state_dict, latent_path, metadata=meta)

            if to_memory:
                # keep it in memory; audio rides inside the latent DTO
                if audio_latent is not None:
                    latent = DTO(latent, audio=audio_latent)
                file_item._encoded_latent = latent.to('cpu')
                if first_frame_latent is not None:
                    file_item._cached_first_frame_latent = first_frame_latent.to('cpu')

            if file_item.is_conversation:
                # the artifact path never binds imgs/frames/tensor
                del latent, state_dict
            else:
                del imgs, latent, frames
                del file_item.tensor
                del state_dict, first_frame_latent, audio_latent
            file_item.cleanup()


class TextEmbeddingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.prompt_embeds: Union[PromptEmbeds, None] = None
        # THE variant memo: one list of EmbedVariant rows replacing the old
        # six per-variant path attributes (caption, DOP pair, the dataset-
        # level shared blank and DOP class-side entries, the per-file D-OPSD
        # teacher pair). Built and gated in get_embed_variants; every
        # consumer (planner, reader, completeness) reads that one table.
        self._embed_variants: Union[list, None] = None
        # diff output preservation embeds (caption with trigger word replaced by class)
        self.dop_prompt_embeds: Union[PromptEmbeds, None] = None
        # D-OPSD teacher embeds (caption with trigger replaced by the self-reference
        # token, encoded WITH the item's own image/video as the vision reference)
        self.dopsd_prompt_embeds: Union[PromptEmbeds, None] = None
        self._loaded_text_embedding_path: Union[str, None] = None
        self._caption_was_dropped = False
        self.is_text_embedding_cached = False
        self.text_embedding_load_device = 'cpu'
        self.text_embedding_version = 1

    def get_text_embedding_info_dict(self: 'FileItemDTO', caption_override=None, text_only=False, dopsd_self_ref=False):
        # make sure the caption is loaded here
        # TODO: we need a way to cache all the other features like trigger words, DOP, etc. For now, we need to throw an error if not compatible.
        if self.caption is None:
            self.load_caption()
        item = OrderedDict([
            ("caption", self.caption if caption_override is None else caption_override),
            ("text_embedding_space_version", self.text_embedding_space_version),
            ("text_embedding_version", self.text_embedding_version),
        ])
        if dopsd_self_ref:
            # teacher embeds carry the item's own media as the vision reference
            item["dopsd_self_ref"] = True
            return item
        # dropout embeds are encoded as plain text, keep control conditioning
        # out of their cache key
        if text_only:
            return item
        # if we have a control image, cache the path
        if self.encode_control_in_text_embeddings and self.control_path is not None:
            item["control_path"] = self.control_path
            # content signature of each control file: a control edited in
            # place must not keep serving the old cached vision blocks (the
            # reference-latent sidecars are content-keyed the same way)
            control_paths = (
                self.control_path
                if isinstance(self.control_path, list)
                else [self.control_path]
            )
            signatures = [get_quick_signature_string(p) for p in control_paths]
            if (
                all(s is not None for s in signatures)
                and (self.dataset_config is None or self.dataset_config.cache_key_on_content)
            ):
                item["control_signatures"] = [_signature_hash(s) for s in signatures]
            if getattr(self, 'text_embedding_uses_target_size', False) and getattr(self, 'crop_width', None):
                item["control_target_size"] = [self.crop_width, self.crop_height]
        if self.encode_control_in_text_embeddings and getattr(self, 'control_video_paths', None):
            item["control_videos"] = sorted(self.control_video_paths)
            # v2: reference-video vision blocks are no longer resampled by the
            # processor (do_sample_frames=False); older video-ref embeds are
            # misaligned with their presentation. Only items WITH control
            # videos carry this key, so no other cache is touched.
            item["control_videos_version"] = 2
        # first-frame vision conditioning changes the embedding content -> new cache key
        elif (
            getattr(self, "encode_first_frame_in_text_embeddings", False)
            and self.dataset_config.do_i2v
            and self.is_video
        ):
            item["first_frame_in_te"] = True
        return item

    def _build_text_embedding_path(self: 'FileItemDTO', caption_override=None, text_only=False,
                                   dopsd_self_ref=False, shared_name=False):
        # we store text embeddings in a folder in same path as image called _text_embedding_cache
        img_dir = os.path.dirname(self.path)
        te_dir = os.path.join(img_dir, '_t_e_cache')
        hash_dict = self.get_text_embedding_info_dict(caption_override=caption_override, text_only=text_only, dopsd_self_ref=dopsd_self_ref)
        # get base64 hash of md5 checksum of hash_dict
        hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
        hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
        hash_str = hash_str.replace('=', '')
        if shared_name:
            # dataset-level blanks are content-addressed: the filename IS the
            # key, with NO image basename, so every file in the folder resolves
            # ONE path per (dropout text, model space). The basename-free name
            # is the sharing contract -- adding a basename prefix here would
            # silently turn the blank back into byte-identical copies per file
            # (docs/refactor_notes/DECISION--blank-embeds-and-dop-gating.md).
            filename_no_ext = f'blank_{hash_str}'
        else:
            # per-caption embeds keep the media basename in the filename (the
            # scheme every existing cache uses -- do not change it, keys churn)
            filename_no_ext = f'{os.path.splitext(os.path.basename(self.path))[0]}_{hash_str}'
        return os.path.join(te_dir, f'{filename_no_ext}.safetensors')

    def get_embed_variants(self: 'FileItemDTO', recalculate=False):
        # THE single table of text-embed variants for this item. Every
        # consumer reads it -- the phase-2 planner (_caption_embed_targets),
        # the train-time reader (load_prompt_embedding), and the completeness
        # gate through the planner -- and none of them gate or dedupe by
        # hand anymore; the gates live here, stated once:
        #   caption      the item caption (control conditioning rides in its
        #                cache key when present, unchanged scheme)
        #   dop          DOP class-side twin of the caption; only under DOP.
        #                If no trigger appears in the caption it hashes onto
        #                the caption path and the file is shared.
        #   blank        the dropout slot: '' (owner contract 2026-10-27),
        #                dataset-level SHARED entry (content-addressed, no
        #                basename -- one file per media folder per model),
        #                plain text only. Planned on every item; identical
        #                paths collapse at the write side.
        #   dop_blank    DOP class side of a DROPPED caption; shared the same
        #                way. Triggerless tokenless items hash it onto '' and
        #                dedupe onto 'blank'.
        #   dopsd        D-OPSD teacher pair (FENCED: per-file by nature --
        #                the vision reference IS the item's own media, so the
        #                basename-prefixed name is deliberate. Do not
        #                "align" these two with the shared blank.)
        #   dopsd_blank  teacher side of a dropped caption, same fencing.
        # The _build_text_embedding_path kwargs are byte-identical to the old
        # per-variant getters', so every existing cache key is unchanged.
        if self._embed_variants is not None and not recalculate:
            return self._embed_variants
        # load the caption first: dop/dopsd texts and inline pairs are built there
        if self.caption is None:
            self.load_caption()
        dc = self.dataset_config

        variants = []

        def add(name, caption, dropout_plain=False, self_ref=False, **build_kwargs):
            variants.append(EmbedVariant(
                name=name,
                path=self._build_text_embedding_path(caption_override=caption, **build_kwargs),
                caption=caption,
                dropout_plain=dropout_plain,
                self_ref=self_ref,
            ))

        add('caption', self.caption)
        if dc.diff_output_preservation:
            add('dop', self.caption_dop)
        if dc.caption_dropout_rate > 0:
            add('blank', self.get_dropout_caption(),
                dropout_plain=True, text_only=True, shared_name=True)
            if dc.diff_output_preservation:
                add('dop_blank', self.get_dop_dropout_caption(),
                    dropout_plain=True, text_only=True, shared_name=True)
        if getattr(self, 'dopsd_self_ref', False):
            add('dopsd', self.caption_dopsd, self_ref=True, dopsd_self_ref=True)
            if dc.caption_dropout_rate > 0:
                add('dopsd_blank', self.get_dopsd_dropout_caption(),
                    self_ref=True, dopsd_self_ref=True)

        self._embed_variants = variants
        return variants

    def get_embed_variant(self: 'FileItemDTO', name, recalculate=False):
        for variant in self.get_embed_variants(recalculate=recalculate):
            if variant.name == name:
                return variant
        return None

    def get_dropout_caption(self: 'FileItemDTO'):
        # Dropout means DROPPED: the text is BLANK_CAPTION_TEXT ('') for
        # every item, always -- the contract block at the top of this module
        # and docs/refactor_notes/DECISION--blank-embeds-and-dop-gating.md.
        # Do not parameterise this by trigger, class, or inline tokens;
        # DOP is a separate mechanism (get_dop_dropout_caption).
        return BLANK_CAPTION_TEXT

    def get_dop_dropout_caption(self: 'FileItemDTO'):
        # DOP is not dropout: class side of the pair when the caption is
        # dropped. Derivation lives in the module-level contract block; the
        # trainer's fixed prior uses dop_prior_fallback_text.
        # Only reachable under DOP -- every caller gates on
        # dataset_config.diff_output_preservation before asking.
        return dop_dropout_class_text(
            getattr(self, '_inline_dop_pairs', None),
            self.trigger_word,
            self.dataset_config.diff_output_preservation_class,
            self.is_reg,
        )

    def get_dopsd_dropout_caption(self: 'FileItemDTO'):
        # D-OPSD-INTERNAL teacher dropout text (owner: leave D-OPSD as it is).
        # This is NOT the dataset caption dropout -- dataset dropout is always
        # '' by contract (get_dropout_caption) and must never route here, and
        # this derivation must never route there. Do not "align" the two.
        pairs = getattr(self, '_inline_dop_pairs', None)
        if pairs is not None and not self.is_reg:
            dropout_caption = ' '.join(dict.fromkeys([t for t, c in pairs]))
        elif self.trigger_word is not None and not self.is_reg:
            dropout_caption = inject_trigger_into_prompt('', trigger=self.trigger_word, add_if_not_present=True)
        else:
            dropout_caption = ''
        if self.trigger_word is not None:
            return dropout_caption.replace(
                self.trigger_word, self.get_dopsd_ref_token()
            )
        return f"{self.get_dopsd_ref_token()} {dropout_caption}".strip()

    def cleanup_text_embedding(self):
        if self.prompt_embeds is not None:
            # we are caching on disk, don't save in memory
            self.prompt_embeds = None
        if self.dop_prompt_embeds is not None:
            self.dop_prompt_embeds = None
        if self.dopsd_prompt_embeds is not None:
            self.dopsd_prompt_embeds = None

    def load_prompt_embedding(self, device=None):
        # train-time reader: consumes the variant table (get_embed_variants)
        # by name -- the SAME table the phase-2 planner wrote from, so a path
        # can never be read differently from how it was written.
        if not self.is_text_embedding_cached:
            return
        if self.prompt_embeds is None:
            self._caption_was_dropped = False
            if self.dataset_config.caption_dropout_rate > 0:
                # get a random float form 0 to 1
                rand = random.random()
                if rand < self.dataset_config.caption_dropout_rate:
                    # drop the caption by using the cached blank embedding
                    # (the dataset-level shared file -- the same path for
                    # every item in the folder)
                    variant = self.get_embed_variant('blank')
                    self._caption_was_dropped = True
            if not self._caption_was_dropped:
                variant = self.get_embed_variant('caption')
            # load it from disk
            self.prompt_embeds = PromptEmbeds.load(variant.path)
            self._loaded_text_embedding_path = variant.path
        if self.dataset_config.diff_output_preservation and self.dop_prompt_embeds is None:
            # DOP is not dropout: a dropped caption still conditions the
            # regularisation prediction on the class side (matches the live
            # path, which routes _caption_was_dropped items through
            # get_dop_dropout_caption). Table names follow that split.
            variant = self.get_embed_variant(
                'dop_blank' if self._caption_was_dropped else 'dop')
            if variant.path == self._loaded_text_embedding_path:
                # no trigger word in caption (or class aliased to the blank):
                # same embedding file
                self.dop_prompt_embeds = self.prompt_embeds
            else:
                self.dop_prompt_embeds = PromptEmbeds.load(variant.path)
        if getattr(self, 'dopsd_self_ref', False) and self.dopsd_prompt_embeds is None:
            # D-OPSD pair is fenced per-file (the item's own media is the
            # vision reference) -- table names, same mechanism
            variant = self.get_embed_variant(
                'dopsd_blank' if self._caption_was_dropped else 'dopsd')
            self.dopsd_prompt_embeds = PromptEmbeds.load(variant.path)

class TextEmbeddingCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.is_caching_text_embeddings = self.dataset_config.cache_text_embeddings

    def _caption_embed_targets(self: 'AiToolkitDataset', file_item):
        # every text-embed target for one item comes from the item's variant
        # table (get_embed_variants): the gates (DOP / dropout / D-OPSD), the
        # path scheme, and the shared-vs-per-file naming are stated THERE,
        # once -- this function only reshapes rows into what phase 2 writes,
        # and the reader + completeness gate see the identical paths.
        # First occurrence of a path wins: two variants that resolve to one
        # file carry byte-equal hash dicts and therefore equal caption text,
        # so encoding either text writes the same bytes (this subsumes the
        # old hand-written per-pair dedupe checks: caption/dop aliasing when
        # no trigger appears, and a triggerless tokenless item's class side
        # aliasing onto the '' blank).
        encode_targets = []
        dropout_target_paths = set()
        dopsd_targets = []
        seen_paths = set()
        for variant in file_item.get_embed_variants(recalculate=True):
            if variant.path in seen_paths:
                continue
            seen_paths.add(variant.path)
            if variant.self_ref:
                # D-OPSD teacher embeds are written WITH the item's own media
                # as the reference -- separate list, per-file names
                dopsd_targets.append((variant.path, variant.caption))
            else:
                encode_targets.append((variant.path, variant.caption))
                if variant.dropout_plain:
                    dropout_target_paths.add(variant.path)
        if dopsd_targets:
            control_video_paths = getattr(file_item, 'control_video_paths', None) or []
            if file_item.control_path is not None or len(control_video_paths) > 0:
                raise ValueError(
                    "D-OPSD self-reference training cannot be combined with "
                    "control images/videos: the item itself must be the only "
                    f"reference. Offending item: {file_item.path}"
                )
        return encode_targets, dropout_target_paths, dopsd_targets

    def _embed_target_valid(self: 'AiToolkitDataset', path: str) -> bool:
        # an embed cache entry counts only if present AND accepted by the
        # model's cache-format check (models that store a derived embed
        # representation reject files from older formats through
        # BaseModel.embed_file_valid; the default accepts anything).
        if not os.path.exists(path):
            return False
        return self.sd.embed_file_valid(path)

    def text_embedding_complete(self: 'AiToolkitDataset'):
        # True when every text embed this dataset would encode already exists
        # on disk, so the training process can skip loading the text encoder.
        if not self.is_caching_text_embeddings:
            return True
        for file_item in self.file_list:
            encode_targets, _, dopsd_targets = self._caption_embed_targets(file_item)
            for path, _ in encode_targets + dopsd_targets:
                if not self._embed_target_valid(path):
                    return False
        return True

    def _batched_plain_text_cache_pass(self: 'AiToolkitDataset'):
        # Phase-2 batched caption caching, opt-in per model
        # (supports_batched_text_cache). Plain-text stale targets -- never
        # control / i2v-first-frame / dopsd items -- are token-counted,
        # packed against the model's measured bytes/padded-token
        # (toolkit/text_cache_batching.py) and encoded one forward per
        # group. Written entries pass _embed_target_valid, so the per-item
        # loop that follows simply skips them; models that do not declare
        # the flag never come here.
        sd = self.sd
        if not getattr(sd, "supports_batched_text_cache", False):
            return
        pairs = []  # (path, caption)
        seen_paths = set()
        for file_item in self.file_list:
            control_video_paths = getattr(file_item, 'control_video_paths', None) or []
            if file_item.encode_control_in_text_embeddings and (
                file_item.control_path is not None or len(control_video_paths) > 0
            ):
                continue
            if (
                getattr(sd, 'encode_first_frame_in_text_embeddings', False)
                and self.dataset_config.do_i2v
                and file_item.is_video
            ):
                continue
            encode_targets, _, _ = self._caption_embed_targets(file_item)
            for path, caption in encode_targets:
                # dedupe by path: the dataset-level blanks are planned once
                # per item but name ONE shared file (targets used to be
                # basename-unique per item, which is why this pass never
                # deduped; without this the shared blank would be packed and
                # encoded once per file again)
                if path in seen_paths or self._embed_target_valid(path):
                    continue
                seen_paths.add(path)
                pairs.append((path, caption))
        if not pairs:
            return
        bpt = int(sd.text_cache_bytes_per_token())
        if bpt <= 0:
            return  # model disabled the pass via model_kwargs
        # the budget is the rest of the card: mem_get_info's free already
        # excludes the resident TE and everything else on the device
        free_bytes, _ = torch.cuda.mem_get_info(sd.device_torch)
        budget_tokens = free_bytes // bpt
        if budget_tokens < 1:
            return
        from toolkit.text_cache_batching import plan_packed_groups, plan_token_groups

        planner = (
            plan_packed_groups
            if getattr(sd, "supports_packed_text_cache", False)
            else plan_token_groups
        )
        counts = sd.count_text_tokens([caption for _, caption in pairs])
        pending = planner(list(zip(counts, pairs)), budget_tokens)
        print_acc(
            f" - Batched caption cache: {len(pairs)} captions -> {len(pending)} groups"
            f" (max bs {max(len(g) for g in pending)}, token budget {budget_tokens}"
            f" @ {bpt} B/token, {free_bytes / 2**30:.1f} GiB free)"
        )
        total = len(pending)
        done = 0
        while pending:
            group = pending.pop(0)  # [(tokens, (path, caption)), ...]
            t0 = time.time()
            try:
                batched = sd.encode_prompt([cap for _, (_, cap) in group])
                parts = type(batched).split_prompt_embeds(batched)
                if len(parts) != len(group):
                    raise RuntimeError(
                        f"batched encode returned {len(parts)} embeds for a "
                        f"group of {len(group)}"
                    )
                for (_, (path, _)), part in zip(group, parts):
                    part.save(path)
                del batched, parts
                dt = time.time() - t0
                done += 1
                tokens = sum(t for t, _ in group)
                print_acc(
                    f"   - group {done}/{total}: {len(group)} captions"
                    f" ({tokens} tokens), {dt:.1f}s"
                    f" ({len(group) / max(dt, 1e-6):.1f} captions/s)"
                )
            except torch.cuda.OutOfMemoryError:
                # the budget is an estimate; splitting keeps the pass moving
                # loudly instead of losing the cache run
                torch.cuda.empty_cache()
                if len(group) == 1:
                    raise
                half = len(group) // 2
                print_acc(f"   - OOM at bs {len(group)}; splitting to {half}+{len(group) - half}")
                pending[0:0] = [group[:half], group[half:]]

    def cache_text_embeddings(self: 'AiToolkitDataset'):
        with accelerator.main_process_first():
            print_acc(f"Caching text_embeddings for {self.dataset_path}")
            print_acc(" - Saving text embeddings to disk")

            # opt-in phase-2 batching; writes land through _embed_target_valid
            # so the per-item loop below skips anything it already saved
            self._batched_plain_text_cache_pass()

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc='Caching text embeddings to disk'):
                file_item.latent_load_device = self.sd.device

                encode_targets, dropout_target_paths, dopsd_targets = self._caption_embed_targets(file_item)
                # only process if not already a valid entry on disk (stale
                # formats re-encode over the old file)
                encode_targets = [
                    t for t in encode_targets if not self._embed_target_valid(t[0])
                ]
                dopsd_targets = [
                    t for t in dopsd_targets if not self._embed_target_valid(t[0])
                ]
                if len(encode_targets) > 0:
                    # text encoder placement is handled by the model's encode_prompt
                    control_video_paths = getattr(file_item, 'control_video_paths', None) or []
                    if file_item.encode_control_in_text_embeddings and (
                        file_item.control_path is not None or len(control_video_paths) > 0
                    ):
                        ctrl_img_list = []
                        control_path_list = file_item.control_path
                        if control_path_list is None:
                            control_path_list = []
                        elif not isinstance(control_path_list, list):
                            control_path_list = [control_path_list]
                        for i in range(len(control_path_list)):
                            try:
                                img = Image.open(control_path_list[i]).convert("RGB")
                                img = exif_transpose(img)
                                # convert to 0 to 1 tensor
                                img = (
                                    TF.to_tensor(img)
                                    .unsqueeze(0)
                                    .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                                )
                                ctrl_img_list.append(img)
                            except Exception as e:
                                print_acc(f"Error: {e}")
                                print_acc(f"Error loading control image: {control_path_list[i]}")
                        # control VIDEOS ride into the presentation by path (models
                        # with supports_video_control_images turn them into
                        # timestamped vision blocks); images first, then videos.
                        # The model needs the dataset config to treat the clip
                        # exactly like its latent rows (frame count / trim)
                        ctrl_img_list.extend(control_video_paths)
                        if len(control_video_paths) > 0:
                            self.sd._ref_video_dataset_config = self.dataset_config
                        
                        if len(ctrl_img_list) == 0:
                            ctrl_img = None
                        elif not self.sd.has_multiple_control_images:
                            ctrl_img = ctrl_img_list[0]
                        else:
                            ctrl_img = ctrl_img_list
                        # the bucket the item trains at, so references can be sized against it
                        target_size = None
                        if getattr(file_item, 'crop_width', None) and getattr(file_item, 'crop_height', None):
                            target_size = (file_item.crop_width, file_item.crop_height)
                        for path, caption in encode_targets:
                            if path in dropout_target_paths:
                                # dropout embeds are plain text. Only fall back to the
                                # control images if the model cannot encode without them
                                try:
                                    prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption)
                                except Exception:
                                    prompt_embeds: PromptEmbeds = self.sd.encode_prompt(
                                        caption, control_images=ctrl_img, target_size=target_size)
                            else:
                                prompt_embeds: PromptEmbeds = self.sd.encode_prompt(
                                    caption, control_images=ctrl_img, target_size=target_size)
                            prompt_embeds.save(path)
                            del prompt_embeds
                    elif (
                        getattr(self.sd, 'encode_first_frame_in_text_embeddings', False)
                        and self.dataset_config.do_i2v
                        and file_item.is_video
                    ):
                        # video item: encode the clip's FIRST FRAME into the text embeddings
                        # as a vision reference, matching sampling (where the ctrl image goes
                        # into the embeds and is held as the clean first frames)
                        file_item.load_and_process_image(self.transform, only_load_latents=True)
                        frames = file_item.tensor  # (T, C, H, W) or (C, H, W), in [-1, 1]
                        first = frames[0] if frames.dim() == 4 else frames
                        ctrl_img = (
                            ((first + 1.0) / 2.0)
                            .clamp(0, 1)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        if self.sd.has_multiple_control_images:
                            ctrl_img = [ctrl_img]
                        for path, caption in encode_targets:
                            if path in dropout_target_paths:
                                # dropout embeds are plain text. Only fall back to the
                                # control images if the model cannot encode without them
                                try:
                                    prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption)
                                except Exception:
                                    prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption, control_images=ctrl_img)
                            else:
                                prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption, control_images=ctrl_img)
                            prompt_embeds.save(path)
                            del prompt_embeds
                        file_item.tensor = None
                    else:
                        for path, caption in encode_targets:
                            prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption)
                            prompt_embeds.save(path)
                            del prompt_embeds
                if getattr(file_item, 'dopsd_self_ref', False):
                    if len(dopsd_targets) > 0:
                        if file_item.is_video:
                            # own path rides through the video-ref presentation
                            ctrl_img = [file_item.path]
                            self.sd._ref_video_dataset_config = self.dataset_config
                        else:
                            # own bucketed pixels as the reference image
                            file_item.load_and_process_image(self.transform, only_load_latents=True)
                            img = file_item.tensor  # (C, H, W) in [-1, 1]
                            ctrl_img = [
                                ((img + 1.0) / 2.0)
                                .clamp(0, 1)
                                .unsqueeze(0)
                                .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                            ]
                            # release the cached latent/pixels the load pulled in
                            file_item.cleanup()
                        if not self.sd.has_multiple_control_images:
                            ctrl_img = ctrl_img[0]
                        for path, caption in dopsd_targets:
                            prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption, control_images=ctrl_img)
                            prompt_embeds.save(path)
                            del prompt_embeds
                file_item.is_text_embedding_cached = True
                i += 1


class CLIPCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.clip_vision_num_unconditional_cache = 20
        self.clip_vision_unconditional_cache = []

    def cache_clip_vision_to_disk(self: 'AiToolkitDataset'):
        if not self.is_caching_clip_vision_to_disk:
            return
        with torch.no_grad():
            print_acc(f"Caching clip vision for {self.dataset_path}")

            print_acc(" - Saving clip to disk")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_clip')

            # make sure the adapter has attributes
            if self.sd.adapter is None:
                raise Exception("Error: must have an adapter to cache clip vision to disk")

            clip_image_processor: CLIPImageProcessor = None
            if hasattr(self.sd.adapter, 'clip_image_processor'):
                clip_image_processor = self.sd.adapter.clip_image_processor

            if clip_image_processor is None:
                raise Exception("Error: must have a clip image processor to cache clip vision to disk")

            vision_encoder: CLIPVisionModelWithProjection = None
            if hasattr(self.sd.adapter, 'image_encoder'):
                vision_encoder = self.sd.adapter.image_encoder
            if hasattr(self.sd.adapter, 'vision_encoder'):
                vision_encoder = self.sd.adapter.vision_encoder

            if vision_encoder is None:
                raise Exception("Error: must have a vision encoder to cache clip vision to disk")

            # move vision encoder to device
            vision_encoder.to(self.sd.device)

            is_quad = self.sd.adapter.config.quad_image
            image_encoder_path = self.sd.adapter.config.image_encoder_path

            dtype = self.sd.torch_dtype
            device = self.sd.device_torch
            if hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero:
                # just to do this, we did :)
                # need more samples as it is random noise
                self.clip_vision_num_unconditional_cache = self.clip_vision_num_unconditional_cache
            else:
                # only need one since it doesnt change
                self.clip_vision_num_unconditional_cache = 1

            # cache unconditionals
            print_acc(f" - Caching {self.clip_vision_num_unconditional_cache} unconditional clip vision to disk")
            clip_vision_cache_path = os.path.join(self.dataset_config.clip_image_path, '_clip_vision_cache')

            unconditional_paths = []

            is_noise_zero = hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero

            for i in range(self.clip_vision_num_unconditional_cache):
                hash_dict = OrderedDict([
                    ("image_encoder_path", image_encoder_path),
                    ("is_quad", is_quad),
                    ("is_noise_zero", is_noise_zero),
                ])
                # get base64 hash of md5 checksum of hash_dict
                hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
                hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
                hash_str = hash_str.replace('=', '')

                uncond_path = os.path.join(clip_vision_cache_path, f'uncond_{hash_str}_{i}.safetensors')
                if os.path.exists(uncond_path):
                    # skip it
                    unconditional_paths.append(uncond_path)
                    continue

                # generate a random image
                img_shape = (1, 3, self.sd.adapter.input_size, self.sd.adapter.input_size)
                if is_noise_zero:
                    tensors_0_1 = torch.rand(img_shape).to(device, dtype=torch.float32)
                else:
                    tensors_0_1 = torch.zeros(img_shape).to(device, dtype=torch.float32)
                clip_image = clip_image_processor(
                    images=tensors_0_1,
                    return_tensors="pt",
                    do_resize=True,
                    do_rescale=False,
                ).pixel_values

                if is_quad:
                    # split the 4x4 grid and stack on batch
                    ci1, ci2 = clip_image.chunk(2, dim=2)
                    ci1, ci3 = ci1.chunk(2, dim=3)
                    ci2, ci4 = ci2.chunk(2, dim=3)
                    clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                clip_output = vision_encoder(
                    clip_image.to(device, dtype=dtype),
                    output_hidden_states=True
                )
                # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                state_dict = OrderedDict([
                    ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                    ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                    ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                ])

                os.makedirs(os.path.dirname(uncond_path), exist_ok=True)
                save_file(state_dict, uncond_path)
                unconditional_paths.append(uncond_path)

            self.clip_vision_unconditional_cache = unconditional_paths

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc=f'Caching clip vision to disk'):
                file_item.is_caching_clip_vision_to_disk = True
                file_item.clip_vision_load_device = self.sd.device
                file_item.clip_vision_is_quad = is_quad
                file_item.clip_image_encoder_path = image_encoder_path
                file_item.clip_vision_unconditional_paths = unconditional_paths
                if file_item.has_clip_augmentations:
                    raise Exception("Error: clip vision caching is not supported with clip augmentations")

                embedding_path = file_item.get_clip_vision_embeddings_path(recalculate=True)
                # check if it is saved to disk already
                if not os.path.exists(embedding_path):
                    # load the image first
                    file_item.load_clip_image()
                    # add batch dimension
                    clip_image = file_item.clip_image_tensor.unsqueeze(0).to(device, dtype=dtype)

                    if is_quad:
                        # split the 4x4 grid and stack on batch
                        ci1, ci2 = clip_image.chunk(2, dim=2)
                        ci1, ci3 = ci1.chunk(2, dim=3)
                        ci2, ci4 = ci2.chunk(2, dim=3)
                        clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                    clip_output = vision_encoder(
                        clip_image.to(device, dtype=dtype),
                        output_hidden_states=True
                    )

                    # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                    state_dict = OrderedDict([
                        ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                        ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                        ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                    ])
                    # metadata
                    meta = get_meta_for_safetensors(file_item.get_clip_vision_info_dict())
                    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
                    save_file(state_dict, embedding_path, metadata=meta)

                    del clip_image
                    del clip_output
                    del file_item.clip_image_tensor

                    # flush(garbage_collect=False)
                file_item.is_vision_clip_cached = True
                i += 1
            # flush every 100
            # if i % 100 == 0:
            #     flush()

        # restore device state
        self.sd.restore_device_state()



class ControlCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
            self.control_generator: ControlGenerator = None
    
    def add_control_path_to_file_item(self: 'AiToolkitDataset', file_item: 'FileItemDTO', control_path: str, gen_type: str):
        if gen_type == 'inpaint':
            file_item.inpaint_path = control_path
            file_item.has_inpaint_image = True
        elif gen_type in ('birefnet', 'sapiens2'):
            file_item.mask_path = control_path
            file_item.has_mask_image = True
        else:
            if file_item.control_path is None:
                file_item.control_path = [control_path]
            elif isinstance(file_item.control_path, str):
                file_item.control_path = [file_item.control_path, control_path]
            elif isinstance(file_item.control_path, list):
                file_item.control_path.append(control_path)
            else:
                raise Exception(f"Error: control_path is not a string or list: {file_item.control_path}")
            file_item.has_control_image = True

    def _generate_images_for_file_items(
            self: 'AiToolkitDataset', generator: 'ControlGenerator', gen_types: List[str], label: str):
        """Generate every requested image type for this dataset's file items,
        wiring the resulting paths on as they are produced. Existing files are
        reused, so this is idempotent across runs."""
        for file_item in tqdm(self.file_list, desc=f'Generating {label}'):
            for gen_type in gen_types:
                gen_path = generator.get_control_path(file_item.path, gen_type)
                if gen_path is not None:
                    self.add_control_path_to_file_item(file_item, gen_path, gen_type)

    def generate_mask_model_files(self: 'AiToolkitDataset', generator: 'ControlGenerator', mask_model: str):
        """One mask model pass for this dataset: generate every mask variant
        the dataset requested from that model and wire the paths. The model
        is already loaded by the caller's pass loop."""
        gen_types = self.mask_model_requests.get(mask_model, [])
        if not gen_types:
            return
        print_acc(f"Generating {'/'.join(gen_types)} masks for {self.dataset_path}")
        self._generate_images_for_file_items(generator, gen_types, f'{mask_model} masks')

    def generate_conditioning_files(self: 'AiToolkitDataset', generator: 'ControlGenerator', control_type: str):
        """One conditioning model pass for this dataset."""
        print_acc(f"Generating '{control_type}' conditioning for {self.dataset_path}")
        self._generate_images_for_file_items(generator, [control_type], f'{control_type} controls')

    def mask_generation_complete(self: 'AiToolkitDataset') -> bool:
        """True when every mask variant this dataset wants already exists on
        disk. This is the phases-guard question for the masking sub-phase."""
        for file_item in self.file_list:
            for gen_types in self.mask_model_requests.values():
                for gen_type in gen_types:
                    if not os.path.exists(generated_image_save_path(file_item.path, gen_type)):
                        return False
        return True

    def conditioning_generation_complete(self: 'AiToolkitDataset') -> bool:
        """Same question for generated conditioning images."""
        for file_item in self.file_list:
            for control_type in self.dataset_config.control_types:
                if not os.path.exists(generated_image_save_path(file_item.path, control_type)):
                    return False
        return True

    def setup_controls(self: 'AiToolkitDataset'):
        """Legacy (non-deferred) path used by generator jobs: run every
        generation pass this dataset wants, one model load per type."""
        wants_masks = any(self.mask_model_requests.values())
        wants_conditioning = len(self.dataset_config.control_types) > 0
        if not wants_masks and not wants_conditioning:
            return
        with torch.no_grad():
            device = self.sd.device
            generator = ControlGenerator(device=device, sd=self.sd)
            try:
                for mask_model in MASK_MODEL_PASS_ORDER:
                    self.generate_mask_model_files(generator, mask_model)
                for control_type in self.dataset_config.control_types:
                    self.generate_conditioning_files(generator, control_type)
            finally:
                generator.cleanup()
            flush()
