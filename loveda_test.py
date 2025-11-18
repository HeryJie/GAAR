import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_GAAR import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, default='config/loveda/GAAR_ufr18.py', help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", default='fig_results/GAAR_ufr18/loveda')
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    # model = Supervision_Train.load_from_checkpoint(
    #     os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path + '.ckpt'), config=config)
    model.cuda()
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset
    if args.val:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
        test_dataset = config.val_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        times = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            torch.cuda.synchronize()
            start_time = time.time()
            raw_predictions = model(input['img'].cuda())
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)
            print(f"mask_real min: {input['gt_semantic_seg'].min().item()}, max: {input['gt_semantic_seg'].max().item()}")
            print(f"predictions min: {predictions.min().item()}, max: {predictions.max().item()}")
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            image_ids = input["img_id"]
            if args.val:
                masks_true = input['gt_semantic_seg']

            img_type = input['img_type']

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                mask_type = img_type[i]
                if args.val:
                    if not os.path.exists(os.path.join(args.output_path, mask_type)):
                        os.mkdir(os.path.join(args.output_path, mask_type))
                    evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                    results.append((mask, str(args.output_path / mask_type / mask_name), args.rgb))
                else:
                    results.append((mask, str(args.output_path / mask_name), args.rgb))
    if args.val:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
            print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
        print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))

    # avg_time = sum(times) / len(times) if times else 0
    # print(f'Average inference time per sample: {avg_time:.4f} seconds')
    # avg_time = sum(times) / len(times) if times else 0
    # fps = 1.0 / avg_time if avg_time > 0 else 0
    # print(f'FPS: {fps:.2f} frames/sec')

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()