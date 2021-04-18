# MARS Pose Annotation Tools: a module for crowdsourcing pose estimation

Before you can fine-tune MARS to your experiment, you need to generate some training data, in the form of manually annotations of animal poses. This directory contains all the tools needed to collect and process manual annotations from a public workforce on Amazon SageMaker Ground Truth.

This involves the following steps:

1. [Extract video frames](#1-extract-video-frames-for-annotation) that you would like to annotate.
2. [Run an AWS labeling job](#2-run-an-annotation-job-on-aws) to collect body part annotations from a human workforce.
3. [Post-process the annotations](#3-post-process-manual-pose-annotations) to correct for common annotation errors.
4. [Visualize some annotations](#4-visualize-some-annotations) and evaluate performance of your workforce.

> **Note**: If you have already collected annotation data from Ground Truth or with the DeepLabCut annotation interface, you can skip this section.

### 1. Extract video frames for annotation
First, we need to collect a set of video frames to annotate. The script `extract_frames.py` will sample frames from all videos found in a directory, and save those frames as jpg files. This script takes the following arguments:

* `input_dir`: directory path to look for video files.
* `output_dir`: directory path to put extracted frames in as jpg files.
* `n_frames`: total number of frames to extract across all the videos.
* `to_skip`: (optional) number of frames to skip at the beginning of each video.

Call it from terminal with:
```python extract_frames.py input_dir '/path/to/videodir' output_dir '/path/to/savedir' n_frames 500 to_skip 100```

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

Follow these [instructions to set up and run annotation jobs](docs/readme_groundTruthSetup.md).

(picture of GT workflow goes here)

### 3. Post-process manual pose annotations
You should now have a folder `output_dir` full of video frames (step 1), and a file called `output.manifest` from your annotation job (step 2). Place a copy of `output.manifest` into `output_dir`. The script `parse_manifest_file.py` will clean up some common annotator errors and save the consolodated data into a pickle file that we'll reference when training the detection and pose models. This script takes the following arguments:

* `data_dir`: directory containing video frames and `output.manifest` from your labeling job.
* `keypoints`: a list of the keypoint names you had annotated; this was set in the `.template` file for your labeling job. If you collected keypoints from multiple mice, just list the body part names here; mouse identifiers are passed in `animal_names` (see below).

It also takes the optional arguments:
* `nWorkers`: (default `5`) the number of workers you had annotate each frame.
* `animal_names`: (default `[]`) animal identifiers used in keypoint names, if keypoints from multiple animals were collected. Eg, ['black','white'] if you collected keypoints from one black mouse and one white mouse.
* `manifest_name`: (default `output.manifest`) name of your manifest file.
* `keyName`: (default `annotatedResult`) name of the manifest field containing annotation data; this was set in Submit_Labeling_Job.ipynb.

Call it from terminal with:
```python parse_manifest_file.py data_dir ... (need to move some of these arguments to a config file)```


### 4. Visualize some annotations
