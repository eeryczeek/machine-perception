import os
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


def prepare_paths(video_id):
    image_path = f"./data/"
    annotation_path = f"./data/ann/"
    output_path = os.path.join("output", video_id)
    os.makedirs(output_path, exist_ok=True)
    return image_path, annotation_path, output_path


def load_mask(annotation_path):
    # Force single-channel (palette) mode
    mask_img = Image.open(os.path.join(annotation_path, "ezgif-frame-001.jpg")).convert(
        "P"
    )
    palette = mask_img.getpalette()
    mask_array = np.array(mask_img)
    objects = np.unique(mask_array)
    objects = objects[objects != 0].tolist()
    return torch.from_numpy(mask_array).long().cuda(), palette, objects


def compute_iou(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / union if union != 0 else 0


def draw_labels(annotation_img, predicted_img, font):
    draw_anno = ImageDraw.Draw(annotation_img)
    draw_pred = ImageDraw.Draw(predicted_img)
    draw_anno.text((10, 10), "Original", fill="white", font=font)
    draw_pred.text((10, 10), "Predicted", fill="white", font=font)


def compose_frame(annotation_img, predicted_img):
    w, h = annotation_img.width + predicted_img.width, annotation_img.height
    combined = Image.new("RGB", (w, h))
    combined.paste(annotation_img.convert("RGB"), (0, 0))
    combined.paste(predicted_img.convert("RGB"), (predicted_img.width, 0))
    return combined


@torch.inference_mode()
@torch.amp.autocast("cuda")
def main():
    print(f"Processing video: eval")
    image_path, annotation_path, output_path = prepare_paths("asdf")
    images = sorted(
        [
            f
            for f in os.listdir(image_path)
            if os.path.isfile(os.path.join(image_path, f))
            and f.lower().endswith((".jpg", ".png"))
        ]
    )
    annotations = sorted(
        [
            f
            for f in os.listdir(annotation_path)
            if os.path.isfile(os.path.join(annotation_path, f))
            and f.lower().endswith((".jpg", ".png"))
        ]
    )

    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    processor.max_internal_size = 480

    mask, palette, objects = load_mask(annotation_path)
    frames, frame_data = [], []

    for ti, (img_name, anno_name) in enumerate(zip(images, annotations)):
        image = Image.open(os.path.join(image_path, img_name))
        tensor = to_tensor(image).cuda().float()

        output = (
            processor.step(tensor, mask, objects) if ti == 0 else processor.step(tensor)
        )
        pred_mask = processor.output_prob_to_mask(output)
        pred_img = Image.fromarray(pred_mask.cpu().numpy().astype(np.uint8))
        pred_img.putpalette(palette)

        frames.append(pred_img)

    gif_path = os.path.join(output_path, "output.gif")
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0
    )
    print(f"GIF saved at {gif_path}")


main()
