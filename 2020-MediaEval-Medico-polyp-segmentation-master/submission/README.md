
# Medico automatic polyp segmentation challenge
This repository aims to provide a skeleton for preparing your submission to the [Medico automatic polyp segmentation challenge](https://multimediaeval.github.io/editions/2020/tasks/medico/).


## How to Participate

We offer two tasks, 1. Polyp Segmentation task , and 2. Alogorithm efficiency task. 

## Task 1. Polyp Segmentation task
For task 1, please submit the prediction masks in a zip file. Please keep the resolution of predicted masks, which is same as the respective test images.

## Task 2. Alogorithm efficiency task 
For, task 2, please submit the Docker image. 

For starters, we require that each submission is delivered in the form of a Docker image. This Docker image should create the submission file described in the challenge description. If you are new to docker, a good place to start is the [Get Started](https://docs.docker.com/get-started/) page of the official Docker documentation.

To participate, we require you to build a Docker image of your submission which includes all required dependencies and can be run using the latest version of Docker. Please note that the data should not be included within the Docker image itself, as it will be injected by us. Assume that the test dataset will be located at `/medico2020`. An example submission is included within this repository, where we show an example of a TesnsorFlow (Keras) based submission.

## Building your Docker image
To build the docker image for the TensorFlow (Keras) based model, please run the following bash command:
```bash
cd submission
sudo docker build -t <docker_id> .
```

## Testing your Docker image
To test you submission, run the following bash command:
```bash
cd submission
docker run -v <test_images_path>:/medico2020 -v <mask_save_path>:/mask <docker_id> > time.txt
```

The time for the test images are saved in the `time.txt` file.

## Submitting your Docker Image
To submit your Docker image, we recommend that you export it using the following bash command:

```bash
sudo docker save <docker_id> > medico2020_image.tar
```

This command will produce a tar file of your Docker image which can easily be sent to one of the organizers of medico2020.

## Example:
```bash
cd submission
sudo docker build -t submission_image.
sudo docker run -v /data/images/:/medico2020 -v /data/mask/:/mask submission_image:latest > time.txt
sudo docker save submission_image > medico2020_image.tar
```

Once the Docker image is exported, submit it to one of the following email addresses; debesh@simula.no, or steven@simula.no. For any questions, feel free to contact us. 
