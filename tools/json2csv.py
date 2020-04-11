import json
import csv

test_json_raw = json.load(open("./data/testB.json", "r"))
annots = json.load(open('./results.json', "r"))

raw_image_filenames = []
images_ids = {}
for img in test_json_raw["images"]:
    images_ids[img["id"]] = img["file_name"]
    raw_image_filenames.append(img["file_name"])
raw_image_filenames = set(raw_image_filenames)

csv_file = open('./data/submission.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(('name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'))

classes = ('holothurian', 'echinus', 'scallop', 'starfish')

for annot in annots:
    name = images_ids[annot['image_id']]
    label = annot["category_id"]
    if label >= 5:
        continue
    score = annot["score"]
    bbox = annot["bbox"]
    w, h = bbox[2], bbox[3]
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + w
    ymax = bbox[1] + h
    writer.writerow((classes[label-1], name.split('.')[0] + '.xml', float(score), round(xmin), round(ymin), round(xmax), round(ymax)))