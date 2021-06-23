
# MARS_Developer
This repository contains all the code you'll need to train your own version of MARS.

## Installation
You will first need to follow the installation instructions for the [end-user version of MARS](https://github.com/neuroethology/MARS).

Once you have the end-user version installed, the only other thing you'll need to do to use `MARS-Developer` is install the conda environment:
```conda env create -f MARS_dev.yml```

You can run jupyter notebooks from within the MARS_dev environment by installing ipykernel:
```
conda install -c anaconda ipykernel
```
and then calling
```
python -m ipykernel install --user --name=mars_dev
```


## The MARS Workflow
MARS processes your videos in three steps:
1) **Detection** - detects the location of animals in each video frame.
2) **Pose estimation** - estimates the posture of each animal in terms of a set of anatomically defined "keypoints".
3) **Behavior classification** - detects social behaviors of interest based on the poses of each animal.

Each of these steps can be fine-tuned to your own data using the code in this repository.

## An Overview of the Training Process
Training MARS to run on your own experiments includes the following steps. We'll assume you have already settled on a recording setup, and have a set of videos on hand to be analyzed.

> **Important Note: the steps below only show the training commands themselves. Open the MARS_tutorial jupyter notebook to see the complete training process, including scripts to evaluate model performance at each step!**

[![](pose_annotation_tools/docs/jupyter_button.jpg)](https://nbviewer.jupyter.org/github/neuroethology/MARS_Developer/blob/master/MARS_tutorial.ipynb)

### 1) 📁 Create a new MARS Training Project.
MARS uses a set file strcture to keep track of data and models associated with your project. The script `create_new_project.py` will create one for you. It takes the arguments:

* `location`: where we're going to save all associated files for this project
* `name`: a name for the project

As well as optional arguments:
* Set `download_MARS_checkpoints=True` to download pre-trained MARS detection and pose models. If your dataset looks similar to MARS, initializing training from the pre-trained models should help decrease your training data requirements. <font color=red>TODO: finish unploading these</font>
* Set `download_demo_data=True` to download a sample dataset, consisting of 2000 frames from videos in the [CRIM13 dataset](https://data.caltech.edu/records/1892), with accompanying manual annotations of animal pose collected via Amazon SageMaker. The sample dataset can also be previewed [here](https://drive.google.com/drive/u/0/folders/1J73k-RC1CyJQOjUdWr-75P3w_mfpRvXr)

Call it from terminal with:
``` python create_new_project.py /path/to/savedir my_project```

It creates a folder at `/path/to/savedir/my_project` that has already been populated with a few files and subdirectories. We'll get to these shortly.

### 2) ✍️ Collect a set of manually annotated animal poses.
We provide code for crowdsourcing of pose annotation to a public workforce via Amazon SageMaker. Running this code requires an Amazon Web Services (AWS) account and some initial time investment in setting up the custom annotation job. A typical pose annotation job, at high annotation quality + high label confidence (5 repeat annotations/image) costs ~68 cents/image.
 - [The MARS Pose Annotation Tools module](pose_annotation_tools#mars-pose-annotation-tools-a-module-for-crowdsourcing-pose-estimation) covers the following steps:
   - [Extracting video frames](pose_annotation_tools#1-extract-video-frames-for-annotation) that you would like to annotate.
   - [Running a labeling job](pose_annotation_tools#2-run-an-annotation-job-on-aws) to collect body part annotations from a human workforce.
   - [Post-processing the annotations](pose_annotation_tools#3-post-process-manual-pose-annotations) to correct for common annotation errors.
   - [Visualizing some annotations](pose_annotation_tools#4-visualize-some-annotations) to evaluate performance of your workforce.

> If you've already collected pose annotations via another interface such as [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/UseOverviewGuide.md#label-frames), you can skip directly to the post-processing step to format your data for training.

After you've collected annotations, the script `pose_annotation_tools.annotation_postprocessing.py` will consolidate annotations across human workers, correct for left/right flips of body parts, and package your data for model training. Call it from the terminal as

```python pose_annotation_tools.annotation_postprocessing.py /path/to/savedir/my_project```

### 3) 🎯 Fine-tune the MARS mouse detector to your data.
Next, we need to teach MARS what your animals look like. The [Multibox Detection module](multibox_detection) covers training, validating, and testing your mouse detector.

To train your detection model(s) from the command line, simply call
``` python multibox_detection/train.py /path/to/savedir/my_project```
This call also takes optional arguments:
* `models` takes names of detectors to train, in case you only want to work on one detector at a time (MARS trains all detectors sequentially by default.)
* `max_training_steps` overrides `NUM_TRAIN_ITERATIONS` in `train_config.yaml` (300,000 by default). You can set this to a small number to confirm everything is working before launching your full training job.

### 4) 🐁 Fine-tune the MARS pose estimator to your data.
Now that you can detect your mice, we want to estimate their poses. In this step we'll train and evaluate a mouse pose estimator for your videos. The [Hourglass Pose module](hourglass_pose) covers training, validating, and testing a stacked hourglass model for animal pose estimation.

As for detection, training the pose model can be launched from the command line by calling
``` python hourglass_pose/train.py /path/to/savedir/my_project```
With optional arguments:
* `models` takes names of pose models to train, in case you only want to work on one model at a time.
* `max_training_steps` overrides `NUM_TRAIN_ITERATIONS` in `train_config.yaml` (300,000 by default).

The module `hourglass_pose/evaluation` includes several scripts for evaluating performance of your pose model using [MARS's fork of the COCO API](https://github.com/neuroethology/MARS_pycocotools). These include `hourglass_pose.evaluation.plot_frame` to view pose predictions for a single frame in your test set, and `hourglass_pose.evaluation.plot_model_PCK` to evaluate performance in terms of Percent Correct Keypoints (PCK).

### 5) 🚀 Deploy your new detection and pose models.
Now that you have a working detector and pose estimator, we'll add them to your end-user version of MARS so you can run them on new videos!
 - Step-by-step: [coming soon]()

### 6) 💪 Train new behavior classifiers.
Once you've applied your trained pose estimator on some new behavior videos, you can annotate behaviors of interest in those videos and train MARS to detect those behaviors automatically.
 - Step-by-step: [coming soon]()
