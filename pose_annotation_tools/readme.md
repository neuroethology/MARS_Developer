# Pose annotation tools

Before you can fine-tune MARS's pose model on your recording setup, you need to generate some training data, in the form of manually annotations of animal poses. This directory contains all the tools needed to collect and process manual annotations from a public workforce on Amazon SageMaker Ground Truth.

This involves the following steps:

1. [Extract video frames](#extract-video-frames-for-annotation) that you would like to annotate.
2. [Run an AWS labeling job](#run-an-annotation-job-on-aws) to collect body part annotations from a human workforce.
3. [Visualize the annotations](#visualize-manual-pose-annotations) from a completed labeling job.
4. [Compile completed pose annotations](#compile-pose-annotations-for-training-mars) so they are ready to use to train MARS.

> **Note**: If you have already collected annotation data from Ground Truth or with the DeepLabCut annotation interface, you can skip straight to **step 4** to compile those annotations for training MARS.

[Click here](MARS_annotation_tools_demo.ipynb) for a Jupyter notebook demo of these steps.

## Extract video frames for annotation
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
## Run an annotation job on AWS
We use Amazon SageMaker Ground Truth to collect manual annotations of animal pose from an annotator workforce. Your annotator workforce can consist of either private annotators (eg lab members), or you can submit jobs to a paid, public workforce.

[Instructions for setting up and running annotation jobs are here](docs/readme_groundTruthSetup.md).

(insert picture of GT workflow)

## Visualize manual pose annotations
Under construction.

## Compile pose annotations for training MARS
Now that you have collected some pose annotations, it's time to package them so you can use them to re-train MARS! To do so, we need to get your annotations into `tfrecord` format, by calling the following from within Python:

```
import MARS_annotation_tools as mat

annotation_file = '/path/to/human_annotations.manifest' # or you can use a csv from DeepLabCut
image_directory = '/path/to/videodir'
keypoint_names = ['Nose','EarL','EarR','Neck','HipL','HipR','Tail']

mat.create_tfrecords(annotation_file, image_directory, keypoint_names)
```
MARS can also handle annotations created using other systems. To make MARS tfrecords from existing DeepLabCut training data, simply set `annotation_file` to be the path to the `.csv` file containing your manual annotations.
