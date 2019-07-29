#                                       RAW PIPELINE OF MARS - NOT OPTIMIZED

In this folder you find and end-to-end framework for MARS.
It is not yet optimized as each step needs to save intermediate files in order be be formatted and passed to next step so
it is still pretty slow overall.
However you can see that running the detection and pose estimation takes around 10ms per image.

**Note**: This pipeline includes, for now, the detection end pose for top view.

All files needed to make it work can be found [here](https://www.dropbox.com/home/team_folder/MARS/pipeline)

### STEPS:
1. Given a seq video file with txt annotation of actions process the video to extract frames and actions, format and save them in tf format
2. Run separate detection
3. Merge in a unique file data from detected bboxes and save them in tf format
4. Run keypoints detection on the merged bboxes

Each step has the option to export frames with the results overimposed on the image

`run_pipeline.sh` is the main file to run in bash to run the pipeline over a set of videos in a folder

We decided to keep separate the bbox detection in order to pass the problem of the identification of the mouse
(black or white). This may need to be taken into account if changing the setting to mice with same coat color

###                                               FILES IN THIS FOLDER

|Name|                                Description|
|------------------------------------|------------|
|run_pipeline.sh                    |   main file to exectute the pipeline|
|model_detection_black.ckpt-200000   |  detection model checkpoint for black mouse|
|model_detection_white.ckpt-200000   |  detection model checkpoint for white mouse|
|priors_gen_black.pkl           |       generated priors for black detection|
|priors_gen_white.pkl                |  generated priors for white detection|
|config_detect_single.yaml          |   configuration file for detection|
|create_ftrecords.py                |   export dictionary to tf format|
|model_detection.py                  |  multibox model for detection|
|video_detection.py                   | detect bbox on video|
|merge_detection.py                  |  merge detection of single mouse to ft format for pose|
|model_pose.ckpt-77288                | pose model checkpoint|
|config_pose.yaml                     | configuration file for pose|
|video_pose.py                       |  detect keypoints on video|
|vis_pose.py                         |  visualize result of the pipeline|
|compute_trajectories.py              | compute trajectories from pose and PNAS features|
|video_maker.py                    |    make video from saved frames|


