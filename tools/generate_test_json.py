import json
import os
from glob import glob
from tqdm import tqdm
import cv2
from PIL import Image
import natsort

def save(images, annotations, name):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    category = [
            {'id': 1, 'name': 'holothurian', 'supercategory': 'water'},
            {'id': 2, 'name': 'echinus', 'supercategory': 'water'},
            {'id': 3, 'name': 'scallop', 'supercategory': 'waterwater'},
            {'id': 4, 'name': 'starfish', 'supercategory': 'water'},
    ]
    ann['categories'] = category
    json.dump(ann, open('./data/{}.json'.format(name), 'w'))


def test_dataset(im_dir):
    im_list = natsort.natsorted(glob(im_dir + '/*.jpg'))
    idx = 1
    image_id = 20190000000
    images = []
    annotations = []
    #h, w, = 1696, 4096
    for im_path in tqdm(im_list):
        #image_id += 1
        if 'template' in os.path.split(im_path)[-1]:
            continue
        #im = cv2.imread(im_path)
        im = Image.open(im_path)
        #h, w = im.[:2]
        w, h = im.size
        image_id += 1
        image = {'file_name': os.path.split(im_path)[-1], 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    save(images, annotations, 'testB')


if __name__ == '__main__':
    # test_dir = '/home/aistudio/image'
    test_dir = '/home/aistudio/work/datasets/water/test-B-image'
    print("generate test json label file.")
    test_dataset(test_dir)