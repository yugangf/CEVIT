import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch import nn
from tqdm import tqdm

import datasets
import dino.vision_transformer as vits
from datasets import Dataset, transform, ImageDataset
from eval import read_annotation, match_detections_to_annotations, calculate_metrics
from object_discovery import canny_edge

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Multi-head-Attention maps")

    parser.add_argument(
        "--arch",
        default="vit_base",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
        ],
        help="Choose which model to be used",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--set",
        default="test",
        type=str,
        choices=["val", "test"],
        help="Path of the image to load",
    )
    # Test does not require annotations, both one image and dataset.
    # Test on one image, please provide the path to the image.
    # Test on dataset, please set "--image_path" to none.

    # Val requires annotations, both one image and dataset.
    # Val on one image, please provide path to both image and xml file in VOC2007 format,
    # please revise these two: "--image_path" and "--anno_path".
    # Val on dataset, please set "--image_path" to none,
    # If you want to use your own dataset, change the path here: "/datasets.py/self.root_path =",
    # and revise "--annos_path" to your annotations
    # Choose to use datasets
    parser.add_argument(
        "--dataset",
        default='mydata',
        type=str,
        choices=["mydata"],
        help="Dataset name",
    )
    parser.add_argument(
        "--annos_path",
        default='D:/Py project/Database/val_separate/val_1500/Annotations',
        # default='D:/Py project/Database/sentinel/Annotations',
        type=str,
        help="Path to annotations of sets ",
    )
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, compute multi-head attention features and save it into save-feat-dir",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default='D:/Py project/Database/val/Images/image_0407.jpg',
        # default=None,
        help="If want to apply only on one image, give absolute path.",
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        # default='D:/Py project/Database/val_separate/val_500/Annotations/image_0407.xml',
        # default=None,
        help="Path to annotation of one image.",
    )
    # Folder used to output visualizations
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory to store predictions and visualizations."
    )
    parser.add_argument("--no_hard", action="store_true",
                        help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument(
        "--ksize",
        type=tuple,
        default=(3, 3),
        choices=[(3, 3), (5, 5)],
        help="GaussianBlur kernel size.",
    )
    # Evaluation setup
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bounding boxes.")
    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["atten", "pred", "all", None],
        default='all',
        help="Select the different type of visualizations.",
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--thres1", type=float, default=30, help="boundary for edge detection.")
    parser.add_argument("--thres2", type=float, default=80, help="both 0 to 200 thres2 must be greater")

    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)
        recall_results = []
        precision_results = []
        f1_score_results = []

    # ------------------------------------------------------------------------------------
    def get_model(arch, patch_size, device):

        # Initialize model with pretraining
        url = None
        if "vit" in arch:
            if arch == "vit_small" and patch_size == 16:
                url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif arch == "vit_small" and patch_size == 8:
                url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            elif arch == "vit_base" and patch_size == 16:
                url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif arch == "vit_base" and patch_size == 8:
                url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        else:
            raise NotImplementedError

        if url is not None:
            print(
                "Loading pretrained DINO weights."
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/" + url
            )
            msg = model.load_state_dict(state_dict, strict=True)
            print(
                "Pretrained weights:{} \nloaded with msg: {}".format(
                    url, msg
                )
            )
        else:
            print(
                "No Pretrained weights available => Use random weights."
            )

        model.eval()
        model.to(device)
        return model


    # ------------------------------------------------------------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, device)
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Experiment with CEVIT
    exp_name = f"ce{args.arch}"
    if "vit" in args.arch:
        exp_name += f"{args.patch_size}"

        print(f"Running CEVIT on the dataset {dataset.name} (model: {exp_name})")
    if args.visualize:
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    if args.save_feat_dir is not None:
        os.mkdir(args.save_feat_dir)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))

    start_time = time.time()
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp)
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded
        # # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():
            if "vit" in args.arch:
                attentions = model.get_last_selfattention(img[None, :, :, :])
                # attentions = model.get_last_selfattention(img.to(device))
                nh = attentions.shape[1]  # number of head
                attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
                attentions = attentions.reshape(nh, w_featmap, h_featmap)
                atten_mean = torch.mean(attentions, dim=0)
                if args.save_feat_dir is not None:
                    np.save(os.path.join(args.save_feat_dir,
                                         im_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png',
                                                                                                          '.npy')),
                            atten_mean.cpu().numpy())
                    continue

            else:
                raise ValueError("Unknown model.")

        object_num, labels, stats, centroids = canny_edge(img, args.patch_size, atten_mean,
                                                          args.ksize, args.thres1, args.thres2)
        font_size = 24
        font = ImageFont.truetype("C:/Windows/Font/pala.ttf", size=font_size)
        text_color = "red"
        if args.image_path is not None:
            detection_result = []
            img = dataset.load_image(im_name, size_im)
            img_result = img.copy()
            scale_factor = args.patch_size

            for i in range(1, object_num):
                x, y, w, h, area = stats[i]
                x_scaled = int(x * scale_factor)
                y_scaled = int(y * scale_factor)
                w_scaled = int(w * scale_factor)
                h_scaled = int(h * scale_factor)

                if area > 1:
                    draw = ImageDraw.Draw(img_result)
                    draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled],
                                   outline="red",
                                   width=2)
                    detection = {"class": "Interference",
                                 "bbox": (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)}
                    detection_result.append(detection)
            if args.set == "test":
                if args.visualize == "pred":
                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
                    img_result.save(save_name1)
                    print(f"Predictions saved at {save_name1}.")


                if args.visualize == "atten":
                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    save_name2 = f"{vis_folder}/{im_name}_Multihead_atten.jpg"
                    plt.axis('off')
                    plt.savefig(f"{vis_folder}/{im_name}_Multihead_atten.jpg", bbox_inches='tight', pad_inches=0.0)
                    print(f"Attention_map saved at {save_name2}.")

                if args.visualize == "all":
                    # img = dataset.load_image(im_name, size_im)
                    # img_result = img.copy()
                    save_name2 = f"{vis_folder}/{im_name}_Multihead_atten.jpg"
                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    plt.axis('off')
                    plt.savefig(save_name2, bbox_inches='tight', pad_inches=0.0)
                    print(f"Attention_map saved at {save_name2}.")

                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
                    img_result.save(save_name1)
                    print(f"Predictions saved at {save_name1}.")
            elif args.set == "val":
                if args.visualize == "atten":
                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    save_name2 = f"{vis_folder}/{im_name}_Multihead_atten.jpg"
                    plt.axis('off')
                    plt.savefig(f"{vis_folder}/{im_name}_Multihead_atten.jpg", bbox_inches='tight', pad_inches=0.0)
                    print(f"Attention_map saved at {save_name2}.")
                if args.visualize == "pred":
                    annotations = read_annotation(args.anno_path)
                    sentences = []
                    detection_result = []
                    for i in range(1, object_num):
                        x, y, w, h, area = stats[i]
                        x_scaled = int(x * scale_factor)
                        y_scaled = int(y * scale_factor)
                        w_scaled = int(w * scale_factor)
                        h_scaled = int(h * scale_factor)

                        if area > 1:
                            draw = ImageDraw.Draw(img_result)
                            draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled],
                                           outline="red",
                                           width=2)
                            detection = {"class": "Interference",
                                         "bbox": (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)}
                            detection_result.append(detection)
                        matched_pairs, iou_matrix = match_detections_to_annotations(detection_result,
                                                                                        annotations['Interference'])

                        for pair in matched_pairs:
                            detection_index, annotation_index = pair
                            iou_score = int(iou_matrix[detection_index][annotation_index] * 100)
                            TP, FP, FN, recall, precision, f1_score = calculate_metrics(matched_pairs,
                                                                                        len(detection_result),
                                                                                        len(annotations[
                                                                                                'Interference']))

                            sentence = f"{detection_result[0]['class']} - {iou_score}%"
                            sentences.append(sentence)
                            detection = detection_result[detection_index]
                            x_scaled, y_scaled = detection['bbox'][:2]

                            text_width, text_height = draw.textsize(sentence, font=font)

                            # 确保文本框不超出图像边界
                            x_scaled = max(0, min(x_scaled, img.width - text_width))
                            y_scaled = max(0, min(y_scaled, img.height - text_height))

                            draw.text((x_scaled, y_scaled - 25), sentence, fill="white", font=font)

                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
                    img_result.save(save_name1)
                    print(f"Predictions saved at {save_name1}.")
                if args.visualize == "all":

                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    save_name2 = f"{vis_folder}/{im_name}_Multihead_atten.jpg"
                    plt.axis('off')
                    plt.savefig(f"{vis_folder}/{im_name}_Multihead_atten.jpg", bbox_inches='tight', pad_inches=0.0)
                    print(f"Attention_map saved at {save_name2}.")

                    annotations = read_annotation(args.anno_path)
                    sentences = []
                    detection_result = []
                    for i in range(1, object_num):
                        x, y, w, h, area = stats[i]
                        x_scaled = int(x * scale_factor)
                        y_scaled = int(y * scale_factor)
                        w_scaled = int(w * scale_factor)
                        h_scaled = int(h * scale_factor)

                        if area > 1:
                            draw = ImageDraw.Draw(img_result)
                            draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled],
                                           outline="red",
                                           width=2)
                            detection = {"class": "Interference",
                                         "bbox": (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)}
                            detection_result.append(detection)
                        matched_pairs, iou_matrix = match_detections_to_annotations(detection_result,
                                                                                        annotations['Interference'])

                        for pair in matched_pairs:
                            detection_index, annotation_index = pair
                            iou_score = int(iou_matrix[detection_index][annotation_index] * 100)
                            TP, FP, FN, recall, precision, f1_score = calculate_metrics(matched_pairs,
                                                                                        len(detection_result),
                                                                                        len(annotations[
                                                                                                'Interference']))

                            sentence = f"{detection_result[0]['class']} - {iou_score}%"
                            sentences.append(sentence)
                            detection = detection_result[detection_index]
                            x_scaled, y_scaled = detection['bbox'][:2]

                            text_width, text_height = draw.textsize(sentence, font=font)

                            # 确保文本框不超出图像边界
                            x_scaled = max(0, min(x_scaled, img.width - text_width))
                            y_scaled = max(0, min(y_scaled, img.height - text_height))

                            draw.text((x_scaled, y_scaled - 25), sentence, fill="white", font=font)

                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
                    img_result.save(save_name1)
                    print(f"Predictions saved at {save_name1}.")
        else:
            detection_result = []
            img = dataset.load_image(im_name)
            img_result = img.copy()
            scale_factor = args.patch_size
            for i in range(1, object_num):
                x, y, w, h, area = stats[i]
                x_scaled = int(x * scale_factor)
                y_scaled = int(y * scale_factor)
                w_scaled = int(w * scale_factor)
                h_scaled = int(h * scale_factor)

                detection = {"class": "Interference",
                             "bbox": (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)}
                detection_result.append(detection)

            if args.set == "test":
                if args.visualize == "atten":
                    save_name1 = f"{vis_folder}/{im_name}_Multi-head_Atten.jpg"
                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    plt.axis('off')
                    plt.savefig(save_name1, bbox_inches='tight', pad_inches=0.0)
                    # print(f"Attention_map saved at {save_name1}.")
                if args.visualize == "pred":
                    fig, ax = plt.subplots()
                    ax.imshow(img_result)
                    for detection in detection_result:
                        x1, y1, x2, y2 = detection["bbox"]
                        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red',
                                         facecolor='none')
                        ax.add_patch(rect)
                    plt.axis('off')
                    save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
                    plt.savefig(save_name1, bbox_inches='tight', pad_inches=0.0)
                    # print(f"Predictions saved at {save_name1}.")
                if args.visualize == "all":
                        save_name1 = f"{vis_folder}/{im_name}_Multi-head_Atten.jpg"
                        plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                        plt.axis('off')
                        plt.savefig(save_name1, bbox_inches='tight', pad_inches=0.0)

                        fig, ax = plt.subplots()
                        ax.imshow(img_result)
                        for detection in detection_result:
                            x1, y1, x2, y2 = detection["bbox"]
                            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red',
                                             facecolor='none')
                            ax.add_patch(rect)
                        plt.axis('off')
                        save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
                        plt.savefig(save_name1, bbox_inches='tight', pad_inches=0.0)

            elif args.set == "val":
                if args.visualize == "atten":
                    save_name1 = f"{vis_folder}/{im_name}_Multi-head_Atten.jpg"
                    plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
                    plt.axis('off')
                    plt.savefig(save_name1, bbox_inches='tight', pad_inches=0.0)
                    # print(f"Attention_map saved at {save_name1}.")
                if args.visualize == "pred":
                    img_result = img.copy()
                    scale_factor = args.patch_size
                    xml_file = os.path.splitext(im_name)[0] + '.xml'
                    xml_path = os.path.join(args.annos_path, xml_file)
                    annotations = read_annotation(xml_path)
                    detection_result = []
                    pil_image = Image.fromarray(img_result)
                    draw = ImageDraw.Draw(pil_image)

                    for i in range(1, object_num):
                        x, y, w, h, area = stats[i]
                        x_scaled = int(x * scale_factor)
                        y_scaled = int(y * scale_factor)
                        w_scaled = int(w * scale_factor)
                        h_scaled = int(h * scale_factor)


                        draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled], outline="red",
                                       width=3)
                        detection = {"class": "Interference",
                                     "bbox": (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)}
                        detection_result.append(detection)

                    matched_pairs, iou_matrix = match_detections_to_annotations(detection_result,
                                                                                annotations['Interference'])
                    # 输出匹配结果
                    sentences = []
                    recall = 0
                    precision = 0
                    f1_score = 0
                    for pair in matched_pairs:
                        detection_index, annotation_index = pair
                        iou_score = int(iou_matrix[detection_index][annotation_index] * 100)
                        TP, FP, FN, recall, precision, f1_score = calculate_metrics(matched_pairs, len(detection_result),
                                                                                    len(annotations['Interference']))

                        sentence = f"{detection_result[0]['class']} - {iou_score}%"
                        sentences.append(sentence)
                        detection = detection_result[detection_index]
                        x_scaled, y_scaled = detection['bbox'][:2]

                        text_width, text_height = draw.textsize(sentence, font=font)

                        # 确保文本框不超出图像边界
                        x_scaled = max(0, min(x_scaled, pil_image.width - text_width))
                        y_scaled = max(0, min(y_scaled, pil_image.height - text_height))
                        draw.text((x_scaled, y_scaled - 25), sentence, fill="white", font=font)
                        # print(
                        #     f"Detection {detection_index} matches Annotation {annotation_index} with IoU {iou_matrix[detection_index][annotation_index]}, Recall {i}: {recall}")
                    recall_results.append(recall)
                    precision_results.append(precision)
                    f1_score_results.append(f1_score)

                    recall_total = sum(recall_results) / len(recall_results)
                    precision_total = sum(precision_results) / len(precision_results)
                    f1_score_total = sum(f1_score_results) / len(f1_score_results)

                    save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
                    pil_image.save(save_name1)












        #         detection_result = []
        #         for i in range(1, object_num):
        #             x, y, w, h, area = stats[i]
        #             x_scaled = int(x * scale_factor)
        #             y_scaled = int(y * scale_factor)
        #             w_scaled = int(w * scale_factor)
        #             h_scaled = int(h * scale_factor)
        #
        #             if area > 1:
        #                 if args.set == "test":
        #                     draw = ImageDraw.Draw(img_result)
        #                     draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled], outline="red",
        #                                    width=2)
        #                     plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
        #                     pltname = f"{vis_folder}/{im_name}_cevit_pred.jpg"
        #                     img_result.save(pltname)
        #                     # print(f"Predictions saved at {pltname}.")
        #                 else:
        #                     annotations = read_annotation(args.anno_path)
        #                     draw = ImageDraw.Draw(img_result)
        #                     draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled], outline="red",
        #                                    width=3)
        #                     detection = {"class": "Interference",
        #                                  "bbox": (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)}
        #                     detection_result.append(detection)
        #
        #                     matched_pairs, iou_matrix = match_detections_to_annotations(detection_result,
        #                                                                                         annotations['Interference'])
        #                     # 输出匹配结果
        #                     # print('\n')
        #                     sentences = []
        #
        #                     recall = 0
        #                     precision = 0
        #                     f1_score = 0
        #                     for pair in matched_pairs:
        #                         detection_index, annotation_index = pair
        #                         iou_score = int(iou_matrix[detection_index][annotation_index] * 100)
        #                         TP, FP, FN, recall, precision, f1_score = calculate_metrics(matched_pairs,
        #                                                                                     len(detection_result),
        #                                                                                     len(annotations[
        #                                                                                             'Interference']))
        #
        #                         sentence = f"{detection_result[0]['class']} - {iou_score}%"
        #                         sentences.append(sentence)
        #                         detection = detection_result[detection_index]
        #                         x_scaled, y_scaled = detection['bbox'][:2]
        #
        #                         text_width, text_height = draw.textsize(sentence, font=font)
        #
        #                         # 确保文本框不超出图像边界
        #                         x_scaled = max(0, min(x_scaled, img.width - text_width))
        #                         y_scaled = max(0, min(y_scaled, img.height - text_height))
        #                         draw.text((x_scaled, y_scaled - 25), sentence, fill="white", font=font)
        #                         # print(
        #                         #     f"Detection {detection_index} matches Annotation {annotation_index} with IoU {iou_matrix[detection_index][annotation_index]}, Recall {i}: {recall}")
        #
        #                 plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
        #                 save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
        #                 img_result.save(save_name1)
        #
        #                 print(f"Predictions saved at {save_name1}.")
        #
        #                 save_name2 = f"{vis_folder}/{im_name}_Multihead_atten.jpg"
        #                 plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
        #                 plt.axis('off')
        #
        #                 plt.savefig(save_name2, bbox_inches='tight', pad_inches=0.0)
        #                 print(f"Attention_map saved at {save_name2}.")
        #             cnt += 1
        #
        # else:
        #     if args.visualize == "atten":
        #         save_name2 = f"{vis_folder}/{im_name}_Multi-head_Atten.jpg"
        #         plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
        #         plt.axis('off')
        #         plt.savefig(save_name2, bbox_inches='tight', pad_inches=0.0)
        #         print(f"Attention_map saved at {save_name2}.")
        #
        #     if args.visualize == "pred":
        #         img = img.cpu()
        #         img1 = (img - img.min()) / (img.max() - img.min()) * 255
        #         img1 = img1.byte()
        #         img = Image.fromarray(img1.permute(1, 2, 0).numpy(), mode='RGB')
        #         img_result = img.copy()
        #         scale_factor = args.patch_size
        #
        #         xml_file = os.path.splitext(im_name)[0] + '.xml'
        #         xml_path = os.path.join(args.annos_path, xml_file)
        #         annotations = read_annotation(xml_path)
        #
        #         detection_result = []
        #         for i in range(1, object_num):
        #             x, y, w, h, area = stats[i]
        #             x_scaled = int(x * scale_factor)
        #             y_scaled = int(y * scale_factor)
        #             w_scaled = int(w * scale_factor)
        #             h_scaled = int(h * scale_factor)
        #             if area > 1:
        #                 # 在原始图像上绘制框
        #                 draw = ImageDraw.Draw(img_result)
        #                 draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled], outline="red",
        #                                width=3)
        #                 detection = {"class": "Interference",
        #                              "bbox": (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)}
        #                 detection_result.append(detection)
        #
        #         matched_pairs, iou_matrix = match_detections_to_annotations(detection_result,
        #                                                                     annotations['Interference'])
        #         # 输出匹配结果
        #         sentences = []
        #         recall = 0
        #         precision = 0
        #         f1_score = 0
        #         for pair in matched_pairs:
        #             detection_index, annotation_index = pair
        #             iou_score = int(iou_matrix[detection_index][annotation_index] * 100)
        #             TP, FP, FN, recall, precision, f1_score = calculate_metrics(matched_pairs, len(detection_result),
        #                                                                         len(annotations['Interference']))
        #
        #             sentence = f"{detection_result[0]['class']} - {iou_score}%"
        #             sentences.append(sentence)
        #             detection = detection_result[detection_index]
        #             x_scaled, y_scaled = detection['bbox'][:2]
        #
        #             text_width, text_height = draw.textsize(sentence, font=font)
        #
        #             # 确保文本框不超出图像边界
        #             x_scaled = max(0, min(x_scaled, img.width - text_width))
        #             y_scaled = max(0, min(y_scaled, img.height - text_height))
        #             draw.text((x_scaled, y_scaled - 25), sentence, fill="white", font=font)
        #             # print(
        #             #     f"Detection {detection_index} matches Annotation {annotation_index} with IoU {iou_matrix[detection_index][annotation_index]}, Recall {i}: {recall}")
        #         recall_results.append(recall)
        #         precision_results.append(precision)
        #         f1_score_results.append(f1_score)
        #
        #         plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
        #
        #         recall_total = sum(recall_results) / len(recall_results)
        #         precision_total = sum(precision_results) / len(precision_results)
        #         f1_score_total = sum(f1_score_results) / len(f1_score_results)
        #
        #         save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
        #         img_result.save(save_name1)
        #     if args.visualize == "all":
        #         img = img.cpu()
        #         img1 = (img - img.min()) / (img.max() - img.min()) * 255
        #         img1 = img1.byte()
        #         img = Image.fromarray(img1.permute(1, 2, 0).numpy(), mode='RGB')
        #         img_result = img.copy()
        #         scale_factor = args.patch_size
        #
        #         xml_file = os.path.splitext(im_name)[0] + '.xml'
        #         xml_path = os.path.join(args.annos_path, xml_file)
        #         annotations = read_annotation(xml_path)
        #
        #         detection_result = []
        #         for i in range(1, object_num):
        #             x, y, w, h, area = stats[i]
        #             x_scaled = int(x * scale_factor)
        #             y_scaled = int(y * scale_factor)
        #             w_scaled = int(w * scale_factor)
        #             h_scaled = int(h * scale_factor)
        #             if area > 1:
        #                 # 在原始图像上绘制框
        #                 draw = ImageDraw.Draw(img_result)
        #                 draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled], outline="red",
        #                                width=3)
        #                 detection = {"class": "Interference",
        #                              "bbox": (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)}
        #                 detection_result.append(detection)
        #
        #         matched_pairs, iou_matrix = match_detections_to_annotations(detection_result,
        #                                                                     annotations['Interference'])
        #         # 输出匹配结果
        #         # print('\n')
        #         sentences = []
        #         recall = 0
        #         precision = 0
        #         f1_score = 0
        #         for pair in matched_pairs:
        #             detection_index, annotation_index = pair
        #             iou_score = int(iou_matrix[detection_index][annotation_index] * 100)
        #             TP, FP, FN, recall, precision, f1_score = calculate_metrics(matched_pairs, len(detection_result),
        #                                                                         len(annotations['Interference']))
        #
        #             sentence = f"{detection_result[0]['class']} - {iou_score}%"
        #             sentences.append(sentence)
        #             detection = detection_result[detection_index]
        #             x_scaled, y_scaled = detection['bbox'][:2]
        #
        #             text_width, text_height = draw.textsize(sentence, font=font)
        #
        #             # 确保文本框不超出图像边界
        #             x_scaled = max(0, min(x_scaled, img.width - text_width))
        #             y_scaled = max(0, min(y_scaled, img.height - text_height))
        #             draw.text((x_scaled, y_scaled - 25), sentence, fill="white", font=font)
        #             # print(
        #             #     f"Detection {detection_index} matches Annotation {annotation_index} with IoU {iou_matrix[detection_index][annotation_index]}, Recall {i}: {recall}")
        #         recall_results.append(recall)
        #         precision_results.append(precision)
        #         f1_score_results.append(f1_score)
        #
        #         plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
        #
        #         recall_total = sum(recall_results) / len(recall_results)
        #         precision_total = sum(precision_results) / len(precision_results)
        #         f1_score_total = sum(f1_score_results) / len(f1_score_results)
        #
        #         save_name1 = f"{vis_folder}/{im_name}_CEVIT_pred.jpg"
        #         img_result.save(save_name1)
        #         # print("\n" f"Predictions saved at {save_name1}.")
        #
        #         save_name2 = f"{vis_folder}/{im_name}_Multi-head_Atten.jpg"
        #         plt.imshow(abs(atten_mean.cpu().numpy()), cmap='cividis')
        #         plt.axis('off')
        #         plt.savefig(save_name2, bbox_inches='tight', pad_inches=0.0)
        #         # print(f"Attention_map saved at {save_name2}.")
        #
        #     cnt += 1
            # if cnt % 50 == 0:
            #     pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")
