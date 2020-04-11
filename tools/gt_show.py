from pycocotools.coco import COCO
import json
import torch
from torch.utils.data import Dataset
import skimage.io
import os
import numpy as np 
import cv2

class CocoDataset(Dataset):

    def __init__(self, root_dir, set_name='train', image_base=''):
        self.root_dir = root_dir
        self.set_name = set_name
        self.image_base = image_base
        self.coco      = COCO(os.path.join(self.root_dir, self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        name = image_info = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
        return img, annot, name

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.image_base, image_info['file_name'])
        img = skimage.io.imread(path)
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

def draw_caption(image, box, caption):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

if not os.path.exists('./output_gt'):
    os.makedirs('./output_gt')

dataset = CocoDataset(root_dir='/home/aistudio/work/datasets/water', set_name='train_aug', image_base='/home/aistudio/work/datasets/water/train/image_aug')
colors = [[np.random.randint(0, 255) for i in range(3)] for i in range(4)]
for i in range(len(dataset)):
    print(i)
    img, annot, name = dataset.__getitem__(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for j in range(annot.shape[0]):
        if annot[j, 4] < 0:
            continue
        bbox = annot[j]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        label_name = dataset.labels[bbox[4]]
        draw_caption(img, (x1, y1, x2, y2), label_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
    cv2.imwrite('output_gt/' + name, img)