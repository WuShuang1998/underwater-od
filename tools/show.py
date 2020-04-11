import json
import mmcv
import numpy as np
import cv2

def draw_caption(image, box, caption):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

annots = json.load(open('results.json'))
img = mmcv.imread('/home/aistudio/work/datasets/water/test-B-image/001200.jpg')
bboxs = []
scores = []
labels = []
names = ['holothurian', 'echinus', 'scallop', 'starfish']
for annot in annots:
    if annot['image_id'] == 20190001200:
        bboxs.append(annot['bbox'])
        scores.append(annot['score'])
        labels.append(annot['category_id'])
for j in range(len(bboxs)):
    score = scores[j]
    if score < 0.3:
        continue
    bbox = bboxs[j]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2]) + x1
    y2 = int(bbox[3]) + y1
    label_name = names[labels[j]-1]
    draw_caption(img, (x1, y1, x2, y2), label_name)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
cv2.imwrite('1.jpg', img)