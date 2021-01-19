
# Auto-Metric Graph Neural Network Based on a Meta-learning Strategy for the Diagnosis of Alzheimer's disease

This repository holds the PyTorch code of our JBHI paper *Auto-Metric Graph Neural Network Based on a Meta-learning Strategy for the Diagnosis of Alzheimer's disease*. 

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes and not for commercial use.

The authors' institution (**Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University**) preserve the copyright and all legal rights of these codes.


## Author List

Xiaofan Song, Mingyi Mao and Xiaohua Qian


## Abstract

Alzheimer's disease (AD) is the most common cognitive disorder. In recent years, many computer-aided diagnosis techniques have been proposed for AD diagnosis and progression predictions. Among them, graph neural networks (GNNs) have received extensive attention owing to their ability to effectively fuse multimodal features and model the correlation between samples. However, many GNNs for node classification use an entire dataset to construct a large fixed-graph structure, which cannot be used for independent testing. To overcome this limitation while maintaining the advantages of the GNN, we propose an auto-metric GNN (AMGNN) model for AD diagnosis. First, a metric-based meta-learning strategy is introduced to realize inductive learning for independent testing through multiple node classification tasks. In the meta-tasks, the small graphs help make the model insensitive to the sample size, thus improving the performance under small sample size conditions. Furthermore, an AMGNN layer with a probability constraint is designed to realize node similarity metric learning and effectively fuse multimodal data. We verified the model on two tasks based on the TADPOLE dataset: early AD diagnosis and mild cognitive impairment (MCI) conversion prediction. Our model provides excellent performance on both tasks with accuracies of 94.44% and 87.50% and median accuracies of 94.19% and 86.25%, respectively. These results show that our model improves flexibility while ensuring a good classification performance, thus promoting the development of graph-based deep learning algorithms for disease diagnosis.


## Requirements

* `pytorch 0.4.0`
* `python 3.7`
* `scikit-learn 0.20.3`


## Contact Information

If you have any questions on this repository or the related paper, feel free to send me an email (xiaohua.qian@sjtu.edu.cn). 


## Citing the Work

If you find our code useful in your research, please consider citing:

```
@journal{song2021jbhi,
    title={Auto-Metric Graph Neural Network Based on a Meta-learning Strategy for the Diagnosis of Alzheimer's disease},
    author={Xiaofan Song, Mingyi Mao and Xiaohua Qian},
    journal={IEEE Journal of Biomedical and Health Informatics}
    month = {January},
    year={2021}
}
```


## Acknowledgements

This code is developed on the code base of [few-shot-gnn](https://github.com/vgsatorras/few-shot-gnn) and [population-gcn](https://github.com/parisots/population-gcn). Many thanks to the authors of these works. 