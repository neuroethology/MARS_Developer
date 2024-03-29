# MARS Pose Annotation Tools: a module for crowdsourcing manual pose annotation

This module will walk you through the process of collecting manual annotations of animal poses, for use in training MARS. This directory contains all the tools needed to collect and process manual annotations from a public workforce on Amazon SageMaker Ground Truth.

This involves the following steps:

1. [Extract video frames](#1-extract-video-frames-for-annotation) that you would like to annotate.
2. [Run an AWS labeling job](#2-run-an-annotation-job-on-aws) to collect body part annotations from a human workforce.
3. [Post-process the annotations](#3-post-process-manual-pose-annotations) to correct for common annotation errors.
4. [Visualize some annotations](#4-visualize-some-annotations) and evaluate performance of your workforce.

### 1. Extract video frames for annotation
First, we need to collect a set of video frames to annotate. The script `extract_frames.py` will sample frames from all videos found in a directory, and save those frames as jpg files. This script takes the following arguments:

* `input_dir`: directory path to look for video files.
* `project`: the path to the labeling project you just created, ie `/path/to/savedir/my_project`.
* `n_frames`: total number of frames to extract across all the videos.
* `to_skip`: (optional) number of frames to skip at the beginning of each video.

Call it from terminal with:
```
python extract_raw_frames.py input_dir /path/to/videodir project /path/to/savedir/my_project n_frames 500 to_skip 100
```
You should now have a folder `my_project/annotation_data` containing a directory `raw_images` of video frames.

### 2. Run an annotation job on AWS
We use Amazon SageMaker Ground Truth to collect manual annotations of animal pose from an annotator workforce. Your annotator workforce can consist of either private annotators (eg lab members), or you can submit jobs to a paid, public workforce.

Follow these [instructions to set up and run annotation jobs](docs/readme_groundTruthSetup.md).

### 3. Post-process manual pose annotations
At the end of step 2, you downloaded a file `output.manifest` of annotation data ([see here for download instructions](docs/readme_groundTruthSetup.md#3-download-the-completed-annotations).)

If you haven't already, copy `output.manifest` to `my_project/annotation_data/output.manifest`. If you name your manifest file something else, or if you want to use annotations from DeepLabCut, edit `my_project/project_config.yaml` and set `manifest_name` to the name of the file containing your annotation data.

Now, we'll use the script `annotation_postprocessing.py` to consolidate the data and clean up some common annotator errors. This script takes the argument:

* `project`: full path to the project directory.

Call it from terminal with:
```python annotation_postprocessing.py project /path/to/savedir/my_project```

### 4. Visualize some annotations
Finally, this module includes some scripts to visualize some of your human pose annotations, inside `pose_annotation_tools/evaluation`. These include:
* `plot_frame(project_path, frame_num)` displays raw worker annotations + the cross-worker median for individual frames
* `plot_human_PCK(project_path)` displays the Percent Correct Keypoint (PCK) curves for individual workers.

The PCK curves show a cdf of the distance from each worker's annotation to the median of the remaining workers, across frames. We plot cdfs of the median worker-to-median distance across frames, as well as a shaded area whose upper and lower bounds are the min and max worker-to-median distances, respectively.

[Check out the Jupyter Notebook Tutorial to learn more about these functions!](../MARS_tutorial.ipynb)
