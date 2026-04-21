"""General-purpose utilities shared across the repo."""

from dateutil import parser
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def clean_datetime_str(date):
    t = date.lower().strip()
    try:
        if "am" in t or "pm" in t:
            pattern = "am" if "am" in t else "pm"
            i0 = t.find(pattern)
            date = t[: i0 + 2].strip()
        elif "gmt" in t:
            i0 = t.find("gmt")
            date = t[:i0].strip()
        else:
            date = t
    except Exception:
        date = None
    return date


def get_clean_date(original_date_str):
    date_str = clean_datetime_str(original_date_str)
    try:
        return parser.parse(date_str)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image cropping / transforms
# ---------------------------------------------------------------------------

tile_size = {"sentinel": 256, "bingmap": 300, "googleEarth": 256, "world_sentinel": 256}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def central_crop_bbox(image_width, image_height, crop_width, crop_height):
    """Bounding box for the central crop of an image: ``(left, upper, right, lower)``."""
    left = (image_width - crop_width) // 2
    upper = (image_height - crop_height) // 2
    right = left + crop_width
    lower = upper + crop_height
    return (left, upper, right, lower)


def get_image(image_path, zoom_level=1, sat_type="bingmap"):
    crop_size = zoom_level * tile_size[sat_type]
    image = Image.open(image_path)
    bbox = central_crop_bbox(
        image_width=image.size[0],
        image_height=image.size[1],
        crop_width=crop_size,
        crop_height=crop_size,
    )
    return image.crop(bbox)


def sat_transform(is_train=True, input_size=224, sat_type="sentinel", zoom_level=1):
    """Build train/eval transforms for satellite imagery."""
    interpol_mode = transforms.InterpolationMode.BICUBIC

    t = []
    # world_sentinel crops already arrive at 256x256; skip the zoom-dependent crop.
    if sat_type != "world_sentinel":
        zl_size = zoom_level * tile_size[sat_type]
        t.append(transforms.CenterCrop(zl_size))
    t.append(transforms.Resize(input_size, interpolation=interpol_mode, antialias=True))
    if is_train:
        t.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD))
    return transforms.Compose(t)
