## Progressive End-to-End Object Detection in Crowded Scenes (Deformable-DETR implementation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](https://github.com/megvii-research/Iter-E2EDET/raw/main/readme/fig.jpg)

## Introduction

In this paper, we propose a new query-based detection framework for crowd detection. Previous query-based detectors suffer from two drawbacks: first, multiple predictions will be inferred for a single object, typically in crowded scenes; second, the performance saturates as the
depth of the decoding stage increases. Benefiting from the nature of the one-to-one label assignment rule, we propose a progressive predicting method to address the above issues. Specifically, we first select accepted queries prone to generate true positive predictions, then refine the rest
noisy queries according to the previously accepted predictions. Experiments show that our method can significantly boost the performance of query-based detectors in crowded scenes. Equipped with our approach, Sparse RCNN achieves 92.0% AP, 41.4% MR^−2 and 83.2% JI on the challenging [CrowdHuman]() dataset, outperforming the box-based method MIP that specifies in handling crowded scenarios. Moreover, the proposed method, robust to crowdedness, can still obtain consistent improvements on moderately and slightly crowded datasets like CityPersons and COCO.

## Models

Experiments of different methods were conducted on CrowdHuman. All approaches take R-50 as the backbone.
Method | #queries | AP | MR | JI 
:---:|:---:|:---:|:---:|:---:
[CrowdDet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chu_Detection_in_Crowded_Scenes_One_Proposal_Multiple_Predictions_CVPR_2020_paper.pdf) | -- | 90.7 | 41.4 | 82.4
Sparse RCNN | 500 | 90.7 | 44.7 | 81.4 
Deformable DETR | 1000 | 91.5 | 43.7 | 83.1
Sparse RCNN + Ours | 500 | 92.0 | 41.4 | 83.2
Deformable DETR + Ours | 1000 | 92.1 | 41.5 | 84.0
## Installation
The codebases are built on top of [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) and [Iter-E2EDET](https://github.com/megvii-research/Iter-E2EDET).

#### Steps
1. Install and build libs following [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).

2. Load the CrowdHuman images from [here](https://www.crowdhuman.org/download.html) and its annotations from [here](https://drive.google.com/file/d/11TKQWUNDf63FbjLHU9iEASm2nE7exgF8/view?usp=sharing). Then update the directory path of the CrowdHuman dataset in the config.py.

3. Train Iter Deformable-DETR
```bash
bash exps/aps.sh
```

4. Evaluate Iter Deformable-DETR. You can download the pre-trained model from [here](https://drive.google.com/file/d/1D8nzWLjZ-eHZG-0pNW0iDm5t9wNVsQSp/view?usp=sharing) for direct evaluation.
```bash
bash exps/aps_test.sh 49
# AP: 0.9216, MR: 0.4140, JI: 0.8389, Recall: 0.9635
```

## License

Iter Deformable-DETR is released under MIT License.


## Citing

If you use our work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX
@article{2022iterdetr,
  title   =  {Progressive End-to-End Object Detection in Crowded Scenes},
  author  =  {Anlin Zheng and Yuang Zhang and Xiangyu Zhang and Xiaojuan Qi and Jian Sun},
  journal =  {arXiv preprint arXiv:arXiv:2203.07669v1},
  year    =  {2022}
}
```
