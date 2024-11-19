# <div style="text-align: center;">DEYOLO: Dual-Feature-Enhancement YOLO for Cross-Modality Object Detection</div>

---

<h4 align="center">Yishuo Chen<sup>1</sup>, Boran Wang<sup>1[✉️]</sup>[wangbr@mail.nankai.edu.cn](mailto:wangbr@mail.nankai.edu.cn), Xinyu Guo<sup>1</sup>, Wenbin Zhu<sup>1</sup>,  
Jiasheng He<sup>1</sup>, Xiaobin Liu<sup>1</sup>  and Jing Yuan<sup>1,2,3</sup> </h4>
<h4 align="center">1.College of Artificial Intelligence, Nankai University </h4>
<h4 align="center">2.Engineering Research Center of Trusted Behavior Intelligence, Ministry of
 Education, Nankai University </h4>
<h4 align="center">3.Tianjin Key Laboratory of Intelligence Robotics, Nankai University </h4>

This repository is the code release of the paper DEYOLO: Dual-Feature-Enhancement YOLO for Cross-Modality Object Detection

---

## Introduction 

We design a dual-enhancement-based cross-modality object
 detection network DEYOLO, in which a semantic-spatial cross-modality
 module and a novel bi-directional decoupled focus module are designed
 to achieve the detection-centered mutual enhancement of RGB-infrared
 (RGB-IR). Specifically, a dual semantic enhancing channel weight assign
ment module (DECA) and a dual spatial enhancing pixel weight assign
ment module (DEPA) are firstly proposed to aggregate cross-modality
 information in the feature space to improve the feature representation
 ability, such that feature fusion can aim at the object detection task.
 Meanwhile, a dual-enhancement mechanism, including enhancements for
 two-modality fusion and single modality, is designed in both DECA and
 DEPA to reduce interference between the two kinds of image modalities.
 Then, a novel bi-directional decoupled focus is developed to enlarge the
 receptive field of the backbone network in different directions, which im
proves the representation quality of DEYOLO.

## Pipeline
### The framework
<div align="center">
  <img src="imgs/network.png" alt="network" width="800" />
</div>

 We incorporate dual-context col
laborative enhancement modules (DECA and DEPA) within the feature extraction
 streams dedicated to each detection head in order to refine the single-modality features
 and fuse multi-modality representations. Concurrently, the Bi-direction Decoupled Fo
cus is inserted in the early layers of the YOLOv8 backbone to expand the network’s
 receptive fields.

### DECA and DEPA
<div align="center">
  <img src="imgs/DECA-DEPA.png" alt="DECA-DEPA" width="800" />
</div>

DECA enhances the cross-modal fusion results by leveraging dependencies between
channels within each modality and outcomes are then used to reinforce the original
single-modal features, highlighting more discriminative channels.  

DEPA is
able to learn dependency structures within and across modalities to produce enhanced
multi-modal representations with stronger positional awareness.

### Bi-direction Decoupled Focus
<div align="center">
  <img src="imgs/bi-focus.png" alt="bi-focus" width="400">
</div>

We divide the pixels into two groups for convolution.
Each group focuses on the adjacent and remote pixels at the same time.
Finally, we concatenate the original feature map in the channel dimension and
make it go through a depth-wise convolution layer.

## Visual comparison
<div align="center">
  <img src="imgs/comparison.png" alt="comparison" width="600" />
</div>

## Main Results
<div align="center">
  <img src="imgs/map.png" alt="map" width="600" />
</div>

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@InProceedings{Chen_2024_ICPR,
    author    = {Chen, Yishuo and Wang, Boran and Guo, Xinyu and Zhu, Senbin and He, Jiasheng and Liu, Xiaobin and Yuan, Jing},
    title     = {DEYOLO: Dual-Feature-Enhancement YOLO for Cross-Modality Object Detection},
    booktitle = {International Conference on Pattern Recognition},
    year      = {2024},
    pages     = {}
}
```

## Acknowledgement
Part of the code is adapted from previous works: [YOLOv8](https://github.com/ultralytics/ultralytics/releases/tag/v8.1.0). We thank all the authors for their contributions.
