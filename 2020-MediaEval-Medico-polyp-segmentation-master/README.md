## 2020 MediaEval Medico - polyp segmentation
The “Medico automatic polyp segmentation task” aims to develop computer-aided diagnosis systems for automatic polyp segmentation to detect all types of polyps (for example, irregular polyp, smaller or flat polyps) with high efficiency and accuracy. The main goal of the challenge is to benchmark semantic segmentation algorithms on a publicly available dataset, emphasizing robustness, speed, and generalization.

Participants will get access to a dataset consisting of 1,000 segmented polyp images from the gastrointestinal tract and a separate testing dataset. The challenge consists of two mandatory tasks, each focused on a different requirement for efficient polyp detection. We hope that this task encourages multimedia researchers to apply their vast knowledge to the medical field and make an impact that may affect real lives.

More information about the challenge can be found on [https://multimediaeval.github.io/editions/2020/tasks/medico/](https://multimediaeval.github.io/editions/2020/tasks/medico/)

## Task Description
The participants are invited to submit the results on the following tasks:

1) Polyp segmentation task (required) - The polyp segmentation task asks participants to develop algorithms for segmenting polyps on a comprehensive dataset.

2) Algorithm efficiency task (required) - The algorithm efficiency task is similar to task one but puts a stronger emphasis on the algorithm’s speed in terms of frames-per-second. To ensure a fair evaluation, this task requires participants to submit a Docker image so that all algorithms are evaluated on the same hardware.

## Training Data
The dataset contains 1,000 polyp images and their corresponding ground truth mask. The datasets were collected from real routine clinical examinations at Vestre Viken Health Trust (VV) in Norway by expert gastroenterologists. The VV is the collaboration of the four hospitals that provide healthcare service to 470,000 peoples. The resolution of images varies from 332✕487 to 1920✕1072 pixels. Some of the images contain green thumbnail in the lower-left corner of the images showing the position marking from the ScopeGuide (Olympus). The training dataset can be downloaded from https://datasets.simula.no/kvasir-seg/.

## Test Data

The test data is released now!!

The data is available at https://drive.google.com/drive/folders/1iZwLDFIh3Z7hn-SOcbbive7Lw4F-XeQ0?usp=sharing

## Evaluation Methodology
The task will use mean Intersection over Union (mIoU) or Jaccard index as an evaluation metric, which is a standard metric for all medical segmentation task. Moreover, we strongly recommend calculating other standard evaluation metrics such as dice coefficient, precision, recall, F2, and frame per second (FPS) for a comprehensive evaluation.

* For task 1 (Polyp segmentation task (required)), please submit the predicted mask in a zip file. 

* For task 2 (Algorithm efficiency task (required)), please submit the docker image.  



## Rules
A participating team will be allowed to make a maximum of 5 submissions. The test data can not be used while training the model. The results will be evaluated by the organizers and presented at MediaEval 2020.

## Task Organizers
* Debesh Jha, SimulaMet debesh (at) simula.no, 
* Steven Hicks, SimulaMet steven (at) simula.no, SimulaMet 
* Michael Riegler, SimulaMet 
* Pål Halvorsen, SimulaMet and OsloMet
* Konstantin Pogorelov, Simula Research Laboratory
* Thomas de Lange, Sahlgrenska University Hospital, Mölndal, Sweden, and Bærum Hospital, Vestre Viken, Norway.

## Contact
Please contact debesh@simula.no, steven@simula.no, michael@simula.no or paalh@simula.no
