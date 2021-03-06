# 2020 MediaEval Medico Challenge: Polyp Segmentation

This repository accompanies our (George Batchkala and Sharib Ali) working-notes paper ["Real-time polyp segmentation using U-Net with IoU loss"](http://ceur-ws.org/Vol-2882/paper30.pdf) presented at [MediaEval 2020
Multimedia Benchmark Workshop](https://eigen.no), which was held online on 14-15 December 2020. If you are interested in this work, we recommend you first getting familiar with the [overview paper](http://ceur-ws.org/Vol-2882/paper1.pdf).

To sum up:
* **Workshop:** [MediaEval 2020 Multimedia Benchmark Workshop](https://eigen.no)
* **Working Notes:** ["Real-time polyp segmentation using U-Net with IoU loss"](http://ceur-ws.org/Vol-2882/paper30.pdf) by George Batchkala and Sharib Ali
* **Joint Publication:** in progress


## About the challenge
**Disclaimer:** next paragraph was directly copied from the official GitHub repository of the challenge: https://github.com/DebeshJha/2020-MediaEval-Medico-polyp-segmentation.

The “Medico automatic polyp segmentation task” aims to develop computer-aided diagnosis systems for automatic polyp segmentation to detect all types of polyps (for example, irregular polyp, smaller or flat polyps) with high efficiency and accuracy. The main goal of the challenge is to benchmark semantic segmentation algorithms on a publicly available dataset, emphasizing robustness, speed, and generalization.

For more Information consult next section (Information and Links).

## Information and Links

- Data: https://datasets.simula.no/kvasir-seg/
- Challenge: https://multimediaeval.github.io/editions/2020/tasks/medico/
- GitHub repository: https://github.com/DebeshJha/2020-MediaEval-Medico-polyp-segmentation

## Repository Contents

- checkpoints:
    - Checkpoints for model weights, best epoch indexes, and validation IoU. More information and the link for download in the README.md inside the folder.

- data:
    - Train (starts download, size 46.2 MB): https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip
    - Test (29.6 MB): https://drive.google.com/file/d/1uP2W2g0iCCS3T6Cf7TPmNdSX4gayOrv2/view?usp=sharing

- models:
    - Different architectures. Each in its own script.

- notebooks:
    - All working notebooks: EDA, experiments.

- presentation:
    - Presentation slides for self-organized mini-conference at the Health Data Science CDT after completing the Deep Learning challenge.

- submission
    - Notebook and script with submission code.
    - Functionality: loads images one by one, makes predictions and records mean number of images/frames processed per second (FPS).

- train-val-split:
    - Copied from the official challenge repository: https://github.com/DebeshJha/2020-MediaEval-Medico-polyp-segmentation/tree/master/kvasir-seg-train-val.
    - Pre-defined train/test split in .txt files (train ids, and validation ids are stored in separate files).

- utils:
    - metrics.py: loss and evaluation metrics
    - dataset.py: custom Dataset class to be used in torch DataLoaders
