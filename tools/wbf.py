import mmcv
import numpy as np
from tqdm import tqdm
import json
import datetime
# from ensemble_boxes import *

def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    for t in range(len(boxes)):
        for j in range(len(boxes[t])):
            label = int(labels[t][j])
            score = scores[t][j]
            if score < thr:
                break
            box_part = boxes[t][j]
            b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]), float(box_part[3])]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse 
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers. 
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model 
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param intersection_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable  
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0 
    
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2). 
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


def get_names(predicted_file):
    names = []
    for predict in predicted_file:
        names.append(predict['image_id'])
    return list(set(names))


if __name__ == "__main__":
    img_infos = mmcv.load('/home/aistudio/work/datasets/water/val.json')['images']
    wh = {}
    for img_info in img_infos:
        wh[img_info['id']] = (img_info['width'], img_info['height'])
    
    CLASSES = ['holothurian', 'echinus', 'scallop', 'starfish']
    model1 = mmcv.load("results_1.json")
    model2 = mmcv.load("results_2.json")
    names1 = get_names(model1)
    names2 = get_names(model2)
    final_names = names1
    num_models = 2
    
    boxes_list = {name: [[] for i in range(num_models)] for name in final_names}
    scores_list = {name: [[] for i in range(num_models)] for name in final_names}
    labels_list = {name: [[] for i in range(num_models)] for name in final_names}
    
    weights = [1, 1]
    iou_thr = 0.7
    skip_box_thr = 0.0001
    sigma = 0.1
    
    for predict in tqdm(model1):
        name = predict['image_id']
        if name not in final_names:
            continue
        cls = predict['category_id']-1
        score = predict['score']
        bbox = predict['bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        bbox[0] /= wh[name][0]
        bbox[1] /= wh[name][1]
        bbox[2] /= wh[name][0]
        bbox[3] /= wh[name][1]
        boxes_list[name][0].append(bbox)
        scores_list[name][0].append(score)
        labels_list[name][0].append(cls)

    for predict in tqdm(model2):
        name = predict['image_id']
        if name not in final_names:
            continue
        cls = predict['category_id']-1
        score = predict['score']
        bbox = predict['bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        bbox[0] /= wh[name][0]
        bbox[1] /= wh[name][1]
        bbox[2] /= wh[name][0]
        bbox[3] /= wh[name][1]
        boxes_list[name][1].append(bbox)
        scores_list[name][1].append(score)
        labels_list[name][1].append(cls)
    
    submit = []
    for name in tqdm(final_names):
        boxes = boxes_list[name]
        score = scores_list[name]
        label = labels_list[name]
        # boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        # boxes_, scores_, labels_ = soft_nms(boxes, score, label, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        # boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes_, scores_, labels_ = weighted_boxes_fusion(boxes, score, label, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes_[:, 2] -= boxes_[:, 0]
        boxes_[:, 3] -= boxes_[:, 1]
        boxes_[:, 0] *= wh[name][0]
        boxes_[:, 1] *= wh[name][1]
        boxes_[:, 2] *= wh[name][0]
        boxes_[:, 3] *= wh[name][1]
        for i in range(len(boxes_)):
            bbox  = boxes_[i]
            res_line = {'image_id': name, 'category_id': int(labels_[i]+1), 'bbox': [float(x) for x in bbox[:4]],
                            'score': float(scores_[i])}
            submit.append(res_line)
    
    print(len(submit))
    out = "./results_merge.json"
    with open(out, 'w') as fp:
        json.dump(submit, fp, indent=4, separators=(',', ': '))
    print('over!')