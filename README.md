# 水下目标检测竞赛
## 代码环境及依赖
* OS：Ubuntu 16.10
* GPU: 1 * 32G-V100
* python：3.7.6
* pytorch：1.1.0
* cudatoolkit：10.0.130

## 训练
```
python tools/train.py configs/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py --gpus 1
python tools/train.py configs/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py --gpus 1
python tools/train.py configs/cascade_rcnn_dconv_c3-c5_r101_fpn_ms800_2000.py --gpus 1
```

## 预测
```
python tools/test.py configs/cascade_rcnn_dcn_x101_64x4d_fpn_1x.py work_dirs/cas_dcn_x101_64x4d_fpn_htc_1x/epoch_4939A.pth --format_only

python tools/test.py configs/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py  work_dirs/cascade_rcnn_dconv_c3-c5_r101_fpn_1x/epoch_4935A.pth --eval bbox

python tools/test.py configs/cascade_rcnn_dconv_c3-c5_r50_fpn_ms800_2000.py  work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_ms800_2000/epoch_12.pth --eval bbox
```

**融合:**
```
python tools/json_merge.py
```
**生成提交文件:**
```
python tools/json2csv.py    #结果保存在./data文件夹中
```