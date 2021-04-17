# Pose annotation tools

Before you can fine-tune MARS's pose model to your recording setup, you need to generate some training data, in the form of manually annotations of animal poses. This directory contains all the tools needed to collect and process manual annotations from a public workforce on Amazon SageMaker Ground Truth.

This involves the following steps:

1. [Extract video frames](#1-extract-video-frames-for-annotation) that you would like to annotate.
2. [Run an AWS labeling job](#2-run-an-annotation-job-on-aws) to collect body part annotations from a human workforce.
3. [Visualize the annotations](#3-visualize-manual-pose-annotations) from a completed labeling job.

> **Note**: If you have already collected annotation data from Ground Truth or with the DeepLabCut annotation interface, you can skip this section.

### 1. Extract video frames for annotation
The script `extract_frames.py` will sample frames from all videos found in a directory, and save those frames as jpg files. This script takes the following arguments:

* `input_dir`: directory path to look for video files.
* `output_dir`: directory path to put extracted frames in as jpg files.
* `n_frames`: total number of frames to extract across all the videos.
* `to_skip`: (optional) number of frames to skip at the beginning of each video.

Call it from terminal with:
`python extract_frames.py input_dir '/path/to/videodir' output_dir '/path/to/savedir' n_frames 500 to_skip 100`

Or use `extract_frames` from within Python by calling:
```
import extract_frames as ef

input_dir = '/path/to/videodir'
output_dir = '/path/to/savedir'
nframes = 1000
to_skip = 100

ef.extract_frames(input_dir, output_dir, n_frames, to_skip)
```
### 2. Run an annotation job on AWS
We use Amazon SageMaker Ground Truth to collect manual annotations of animal pose from an annotator workforce. Your annotator workforce can consist of either private annotators (eg lab members), or you can submit jobs to a paid, public workforce.

* First, follow these [instructions for setting up and running annotation jobs](docs/readme_groundTruthSetup.md).
* Next, open the [MARS_annotation_tools_demo](MARS_annotation_tools_demo.ipynb) Jupyter notebook from a SageMaker Notebook Instance to launch your labeling job.

(picture of GT workflow goes here)

### 3. Visualize manual pose annotations
Under construction.
