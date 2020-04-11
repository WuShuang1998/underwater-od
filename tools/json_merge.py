import mmcv
import numpy as np
from tqdm import tqdm
from mmdet.ops.nms import nms_wrapper
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import json
import datetime


def get_names(predicted_file):
    names = []
    for predict in predicted_file:
        names.append(predict['image_id'])
    return list(set(names))

def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]

    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out

if __name__ == "__main__":
    nms_op = nms_wrapper.soft_nms
    CLASSES = ['holothurian', 'echinus', 'scallop', 'starfish']
    model1 = mmcv.load("results_x101.json")
    model2 = mmcv.load("results_r101.json")
    model3 = mmcv.load("results_r101_nonlocal.json")
    names1 = get_names(model1)
    names2 = get_names(model2)
    names3 = get_names(model3)
    final_names = names1

    result = {name: [[] for i in range(len(CLASSES))] for name in final_names}
    for predict in tqdm(model1):
        name = predict['image_id']
        if name not in final_names:
            continue
        cls = predict['category_id']-1
        if cls >= 4:
            continue
        score = predict['score']
        bbox = predict['bbox']
        bbox = bbox + [score]
        result[name][cls].append(np.array(bbox))
        
    for predict in tqdm(model2):
        name = predict['image_id']
        if name not in final_names:
            continue
        cls = predict['category_id']-1
        if cls >= 4:
            continue
        score = predict['score']
        bbox = predict['bbox']
        bbox = bbox + [score]
        result[name][cls].append(np.array(bbox))
    
    for predict in tqdm(model3):
        name = predict['image_id']
        if name not in final_names:
            continue
        cls = predict['category_id']-1
        if cls >= 4:
            continue
        score = predict['score']
        bbox = predict['bbox']
        bbox = bbox + [score]
        result[name][cls].append(np.array(bbox))
    
    submit = []
    for name in tqdm(final_names):
        for i in range(len(CLASSES)):
            det = np.array(result[name][i])
            if len(det) == 0:
                continue
            det[:, 2] += det[:, 0]
            det[:, 3] += det[:, 1]
            if det.shape[0] == 0:
                continue
            cls_dets, _ = nms_op(det, iou_thr=0.7)
            # cls_dets = box_voting(np.array(cls_dets, dtype=np.float32), np.array(det, np.float32), thresh=0.5,
            #                       scoring_method='IOU_AVG', beta=1.0)
            for bbox in cls_dets:
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                res_line = {'image_id': name, 'category_id': int(i+1), 'bbox': [float(x) for x in bbox[:4]],
                            'score': float(bbox[4])}
                submit.append(res_line)
    print(len(submit))
    print(len(final_names))
    out = "./results.json"
    with open(out, 'w') as fp:
        json.dump(submit, fp, indent=4, separators=(',', ': '))
    print('over!')