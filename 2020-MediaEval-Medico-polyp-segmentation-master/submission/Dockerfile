
# Use an intermidate Docker image as base for our submissioin.
# Replace this with whatever is appropriate for your submission.
FROM tensorflow/tensorflow:2.2.0

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

# Install required pip packages
RUN pip3 install pillow
RUN pip3 install numpy
RUN pip3 install opencv-python
RUN pip3 install pandas

# Copy the contents of this directory (submission_1)
# to the /submission path of the Docker image.
ADD . /submission

# Run the "submission_1.py" python file.
# For your submission, replace with your test script.
CMD [ "python3", "/submission/submission.py" ]
