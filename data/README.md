# Data

1. Download the data and rename folders;
2. Unzip the data;
2. Move into `train-val` and `test`.

## Directory Contents

Download the datasets for training/validation and the competition test set (no labels publicly available).

- Training and validation data: (starts download, size 46.2 MB): https://datasets.simula.no/downloads/kvasir-seg.zip

- Test data (29.6 MB): https://drive.google.com/file/d/1uP2W2g0iCCS3T6Cf7TPmNdSX4gayOrv2/view?usp=sharing


```bash
# unzip and move training/validation data
unzip kvasir-seg.zip
mv Kvasir-SEG/ train-val

# unzip and move test data
unzip Medico_automatic_polyp_segmentation_challenge_test_data.zip
mv Medico_automatic_polyp_segmentation_challenge_test_data/ test
```