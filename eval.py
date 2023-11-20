import math
import xml.etree.ElementTree as ET

import torch


def read_annotation(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    annotation_data = {}

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        if class_name in annotation_data:
            annotation_data[class_name].append((xmin, ymin, xmax, ymax))
        else:
            annotation_data[class_name] = [(xmin, ymin, xmax, ymax)]

    return annotation_data


def calculate_iou(bbox1, bbox2):
    # 提取坐标
    x1_detec, y1_detec, x2_detec, y2_detec = bbox1
    x1_anno, y1_anno, x2_anno, y2_anno = bbox2

    # 计算交集的坐标
    x_intersection = max(x1_detec, x1_anno)
    y_intersection = max(y1_detec, y1_anno)
    x_intersection_end = min(x2_detec, x2_anno)
    y_intersection_end = min(y2_detec, y2_anno)

    # 计算交集的宽度和高度
    intersection_width = max(0, x_intersection_end - x_intersection)
    intersection_height = max(0, y_intersection_end - y_intersection)

    # 计算交并比
    det_area = (x2_detec - x1_detec) * (y2_detec - y1_detec)
    anno_area = (x2_anno - x1_anno) * (y2_anno - y1_anno)
    intersection_area = intersection_width * intersection_height

    iou = intersection_area / (det_area + anno_area - intersection_area)

    return iou


def match_detections_to_annotations(detection_list, annotation_list):
    iou_matrix = []

    for detection in detection_list:
        iou_values = []

        for annotation in annotation_list:
            iou = calculate_iou(detection['bbox'], annotation)
            iou_values.append(iou)

        iou_matrix.append(iou_values)
    matched_pairs = []

    for i in range(len(detection_list)):
        max_iou = max(iou_matrix[i])
        if max_iou > 0.5:  # 设置一个IoU阈值来确定匹配
            max_iou_index = iou_matrix[i].index(max_iou)
            matched_pairs.append((i, max_iou_index))

    return matched_pairs, iou_matrix


# 计算性能指标
def calculate_metrics(matched_pairs, detection_count, annotation_count):
    TP = len(matched_pairs)
    FP = detection_count - TP
    FN = annotation_count - TP

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return TP, FP, FN, recall, precision, f1_score
