# MARS Pose Annotation Tools: a module for crowdsourcing manual pose annotation

This module will walk you through the process of collecting manual annotations of animal poses, for use in training MARS. This directory contains all the tools needed to collect and process manual annotations from a public workforce on Amazon SageMaker Ground Truth.

This involves the following steps:

0. [Initialize a new labeling project](#0-initialize-a-new-labeling-project), creating a project directory that will organize all your files for an annotation job.
1. [Extract video frames](#1-extract-video-frames-for-annotation) that you would like to annotate.
2. [Run an AWS labeling job](#2-run-an-annotation-job-on-aws) to collect body part annotations from a human workforce.
3. [Post-process the annotations](#3-post-process-manual-pose-annotations) to correct for common annotation errors.
4. [Visualize some annotations](#4-visualize-some-annotations) and evaluate performance of your workforce.

### 0. Initialize a new labeling project
Before we go any further, we're going to create a master directory for all files associated with this labeling job. This will make life easier when it comes to training MARS down the line. The script `new_annotation_project.py` will get you started. It takes the arguments:

* `location`: where we're going to save all associated files for this project
* `name`: a name for the project

Call it from terminal with:
``` python new_annotation_project.py location '/path/to/savedir' name 'my_project'```

And you'll see that it's created a new folder at `/path/to/savedir/my_project`. Inside this folder is a subdirectory called `annotation_data` that has already been populated with a few files. We'll get to those shortly.

### 1. Extract video frames for annotation
Next, we need to collect a set of video frames to annotate. The script `extract_frames.py` will sample frames from all videos found in a directory, and save those frames as jpg files. This script takes the following arguments:

* `input_dir`: directory path to look for video files.
* `project_path`: the path to the labeling project you just created, ie `/where/to/save/data/my_project`. **TODO: create wrapper for extract_frames that uses the project_path flag and also creates a manifest??**
* `n_frames`: total number of frames to extract across all the videos.
* `to_skip`: (optional) number of frames to skip at the beginning of each video.

Call it from terminal with:
```
python extract_frames.py input_dir '/path/to/videodir' project_path '/path/to/savedir/my_project' n_frames 500 to_skip 100
```

Or use `extract_frames` from within Python by calling:
```
import extract_frames as ef

input_dir = '/path/to/videodir'
project_path = '/path/to/savedir/my_project'
nframes = 1000
to_skip = 100

ef.extract_frames(input_dir, project_path, n_frames, to_skip)
```
### 2. Run an annotation job on AWS
We use Amazon SageMaker Ground Truth to collect manual annotations of animal pose from an annotator workforce. Your annotator workforce can consist of either private annotators (eg lab members), or you can submit jobs to a paid, public workforce.

Follow these [instructions to set up and run annotation jobs](docs/readme_groundTruthSetup.md).

(picture of GT workflow goes here)

### 3. Post-process manual pose annotations
You should now have a folder `my_project/annotation_data` containing video frames (step 1), the `annot_config.yml` file that you edited while setting up your annotation job (step 2), and your annotation .template file (step 2).

Now, place a copy of `output.manifest` into `my_project/annotation_data` ((see here)[docs/readme_groundTruthSetup.md#5-downloading-the-completed-annotations] for download instructions).

We'll next use the script `parse_manifest_file.py` to clean up some common annotator errors and save the consolodated data into a pickle file that we'll reference when training the detection and pose models. This script takes the following arguments:

* `data_dir`: directory containing video frames, `config.yml`, and `output.manifest` from your labeling job.

It also takes the optional arguments:
* `nWorkers`: (default `5`) the number of workers you had annotate each frame.
* `manifest_name`: (default `output.manifest`) name of your manifest file.
* `keyName`: (default `annotatedResult`) name of the manifest field containing annotation data; this was set in Submit_Labeling_Job.ipynb.

Call it from terminal with:
```python parse_manifest_file.py data_dir ... (need to move some of these arguments to a config file)```


### 4. Visualize some annotations
Now we're going to run a script that will a) show some example annotated images from your dataset, and b) save some summary statistics on annotator performance, so you can evaluate how well people labeled your data. Woohoo.
