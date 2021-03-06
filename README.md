Context-Aware Graph Inference with Knowledge Distillation for Visual Dialog
====================================


![alt text](https://github.com/wh0330/CAG_VisDial/blob/master/image/framework.png)
<p align="center">The overall framework of Context-Aware Graph.</p>


![alt text ](https://github.com/wh0330/VisDial_CAG_Distill/blob/main/image/visdial_distill.png)
<p align="center">Knowledge distillation between CAG and Img-Only models.</p>



This is a PyTorch implementation for Context-Aware Graph Inference with Knowledge Distillation for Visual Dialog.


If you use this code in your research, please consider citing:


Requirements
----------------------
This code is implemented using PyTorch v0.3.1, and provides out of the box support with CUDA 9 and CuDNN 7. 


Data
----------------------

1. Download the VisDial v1.0 dialog json files and images from [here][1].
2. Download the word counts file for VisDial v1.0 train split from [here][2]. 
3. Use Faster-RCNN to extract image features from [here][3].
4. Download pre-trained GloVe word vectors from [here][4].
5. We collected a specific subset from Visdial v1.0 val, called Visdial v1.0 (val-yn) (mentioned in our paper) in the folder [subdataset](https://github.com/wh0330/VisDial-CAG-Distill/tree/main/subdataset).


Pre-train
--------

Train the CAG model as:
```sh
python train/train.py --cuda --encoder=CAGraph
```

Train the Img-Only model as:
```sh
python train/train.py --cuda --encoder=Img_only
```
Distillation
--------

First, use the pre-trained Img-only model to generate soft-labels:
```sh
python train/soft_labels.py --model_path [path_to_root]/save/pretrained_img_only.pth --cuda
```
Then, fine-tune the pre-trained CAG model as:
```sh
python train/train_distill.py --model_path [path_to_root]/save/pretrained_cag.pth  --softlabel ./soft_labels.h5 --cuda
```
Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```sh
python eval/evaluate.py --model_path [path_to_root]/save/XXXXX.pth --cuda
```
This will generate an EvalAI submission file, and you can submit the json file to [online evaluation server][5] to get the result on v1.0 test-std.

  Model  |  NDCG   |  MRR   |  R@1  | R@5  |  R@10   |  Mean  |
 ------- | ------ | ------ | ------ | ------ | ------ | ------ |
CAG | 56.64 | 63.49 | 49.85 |  80.63| 90.15 | 4.11 |
CAG-Distill | 57.77 | 64.62 | 51.28 |  80.58| 90.23 | 4.05 |

Acknowledgements
----------------

* This code began with [jiasenlu/visDial.pytorch][6]. We thank the developers for doing most of the heavy-lifting.


[1]: https://visualdialog.org/data
[2]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json
[3]: https://github.com/peteanderson80/bottom-up-attention
[4]: https://github.com/stanfordnlp/GloVe
[5]: https://evalai.cloudcv.org/web/challenges/challenge-page/161/overview
[6]: https://github.com/jiasenlu/visDial.pytorch
