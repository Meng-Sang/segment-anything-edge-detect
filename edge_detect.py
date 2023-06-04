import argparse
import os
import sys

import cv2
import numpy as np
import tqdm

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def load_model(sam_checkpoint, model_type, device):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.7,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    return mask_generator


def predict(net, image):
    masks = net.generate(image)
    return masks


def get_edge_from_masks(masks):
    edge_image = np.zeros(masks[0]["segmentation"].shape)
    for mask in masks:
        gray_image = (np.array(mask["segmentation"]) * 255).astype(np.uint8)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, binary_image = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)
        edge_image += binary_image
    return edge_image


def main(edge_dir, save_dir, model_args):
    net = load_model(model_args.sam_checkpoint, model_args.model_type, model_args.device)
    image_name_list = os.listdir(edge_dir)
    if not os.path.exists(edge_dir):
        raise Exception(f"{edge_dir} not exist")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for image_name in tqdm.tqdm(image_name_list):
        image = cv2.imread(os.path.join(edge_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = predict(net, image)
        edge_image = get_edge_from_masks(masks)
        cv2.imwrite(os.path.join(save_dir, image_name), edge_image)
    print("Finished !!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Generation Edge",
                                     description="Generate Edge Image with Segment Anything and Sobel",
                                     allow_abbrev=True)
    #
    parser.add_argument("--edge_dir", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--sam_checkpoint", default="./weight/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--model_type", default="vit_h", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    main(args.edge_dir, args.save_dir, args)
