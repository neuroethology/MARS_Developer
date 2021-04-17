
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

## Master Tutorial
If you would like to train MARS to run on your own experiments, you will need to carry out the following steps. We'll assume you have already settled on a recording setup, and have a set of videos on hand to be analyzed.

### 1) ✍️ Collect a set of manually annotated animal poses.
We provide code for crowdsourcing of pose annotation to a public workforce via Amazon SageMaker. Running this code requires an AWS account and some initial time investment in setting up the custom annotation job. A typical pose annotation job, at high annotation quality + high label confidence (5 repeat annotations/image) costs ~68c/image.
 - [The MARS Pose Annotation Tools module](https://github.com/neuroethology/MARS_Developer/tree/develop/pose_annotation_tools) covers the following step-by-step instructions:
   -  1.1 [Extract video frames for annotation](https://github.com/neuroethology/MARS_Developer/tree/develop/pose_annotation_tools#1-extract-video-frames-for-annotation).
   -  1.2 [Run an annotation job on AWS](https://github.com/neuroethology/MARS_Developer/tree/develop/pose_annotation_tools#2-run-an-annotation-job-on-aws).
   -  1.3 [Visualize your manual pose annotations](https://github.com/neuroethology/MARS_Developer/tree/develop/pose_annotation_tools#3-visualize-manual-pose-annotations).

### 2) 📁 Create a new MARS Training Project.
Now that you've collected some pose annotation data (either via AWS or from another pose annotation interface such as [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/UseOverviewGuide.md#label-frames)), we will set up a new Project folder containing all the files you'll need to train a new MARS detector or pose estimator.
 - Step-by-step: [something]()

### 3) 🎯 Fine-tune the MARS mouse detector to your data.
First, we need to teach MARS what your animals look like. In this step, we'll create a mouse detector for your videos, and check its performance.
 - Step-by-step: [something]()

### 4) 🐁 Fine-tune the MARS pose estimator to your data.
Now that you can detect your mice, we want to estimate their poses. In this step we'll train and evaluate a mouse pose estimator for your videos.
 - Step-by-step: [something]()

### 5) 🚀 Deploy your new detection and pose models.
Now that you have a working detector and pose estimator, we'll add them to your end-user version of MARS so you can run them on new videos!
 - Step-by-step: [something]()

### 6) 💪 Train new behavior classifiers.
Once you've applied your trained pose estimator on some new behavior videos, you can annotate behaviors of interest in those videos and train MARS to detect those behaviors automatically.
 - Step-by-step: [something]()
