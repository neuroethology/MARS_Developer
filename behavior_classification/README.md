## Overview
I've included the following scripts here:
- [MARS_feature_extractor.py](##MARS_feature_extractor.py): for extracting custom subsets of features from a pose `*.json`.
- [MARS_clf_helpers.py](##MARS_clf_helpers.py): for evaluating classifier performance. 
- [MARS_ts_util.py](##MARS_ts_util.py): scripts for smoothing/windowing features or convolving them with wavelets.

## MARS_feature_extractor.py
Scripts for extracting features from mouse poses, to experiment with feature engineering for improved classification.

**How it works**: I've grouped MARS's pose-derived features into categories (position-based features, orientation-based features, etc). To get a list of these categories, and the features included in each category, call `list_features()`. Given a pose file at `pose_fullpath`, calling `extract_features(pose_fullpath, use_grps=['<category name>','<category name>',...])` returns a **dict** with just the features in those categories, organized in the same way as our `*.npz` feature files (details on **dict** structure below).

If you wanted to make a new feature group not included in any of the above, you can add it to the code of `generate_feature_list()`. Eg, say you wanted to train a classifier using just the speed and distance to walls of mouse 1, you would add the following line to `generate_feature_list()`:

`feats['top']['m1']['my_custom_group'] = ['speed_centroid','dist_edge']`

Then you could generate just those features by calling `my_feats = extract_features(path_to_pose_json, use_grps=['my_custom_group'])`.

### `extract_features(pose_fullpath, use_grps=[], smooth_keypoints=False, center_mouse=False)`
Given the path to a pose file (json), this function returns a requested set of extracted features.

##### Inputs:
* `pose_fullpath` **(string)**: the absolute path to a `*.json` pose file.
* `use_grps = []` **(string array)**: which feature categories to extract- if empty, extracts all of them. To get a list of available feature categories, call `list_features()`.
* `smooth_keypoints = (False)|True`: if true, will try to denoise keypoint trajectories by calling `smooth_keypoint_trajectories()` before extracting features. **Smoothing not yet implemented** so this currently does nothing!
* `center_mouse = (False)|True`: if true, will rotate+translate keypoints to an egocentric coordinate frame by calling `center_on_mouse()` before features are extracted. This puts the neck of the Resident mouse (mouse 1) at the origin, and rotates so the line between the ears is horizontal. Could help get rid of un-interesting sources of variance in the data, but is also sensitive to noise in neck/ear keypoint estimates (so smoothing first is probably important.)

##### Outputs:
Returns a **dict** with the following fields:
* `vid_name` **(string)**: the path passed as input.
* `features` **(string array)**: names of all the features.
* `data` **(numpy array)**: (2 x time x num_features) array of extracted features!
* `bbox` **(numpy array)**:  bounding boxes inherited from the `pose_fullpath` file.
* `keypoints` **(numpy array)**: keypoints inherited from the `pose_fullpath` file.
* `fps` **(double)**: framerate of the movie in `vid_name`, if provided- otherwise assumes 30Hz

### `list_features()`
Prints a list of all available feature groups, as well as a breakdown of which features are in each group. I'm still working on documentation explaining what each of the features are (there are a lot of them) but you can get an idea by looking at the lambdas created for each feature, in the function `generate_lambdas()`. Features that are the same for mouse 1 vs mouse 2, like nose-nose distance, are only defined for mouse 1. You can ignore front-camera features for now.

## MARS_clf_helpers.py

#### `prf_metrics(y_tr_beh, pd_class, beh)`
Computes precision and recall for classifier output, given numpy arrays of 0's and 1's (representing presence of a behavior).

##### Inputs:
* `y_tr_beh` **(1D binary numpy array)**: ground truth annotations
* `pd_class` **(1D binary numpy array)**: predicted labels
* `beh` **(string)**: name of the behavior being detected

##### Output:

Prints precision (P), recall (R), and F1 score (F1) to the command line.

## MARS_ts_util.py
A handful of functions for cleaning up or transforming pose features. Most of these are for a step in existing MARS classifiers where annotations are cleaned up using forward/backward smoothing and/or an HMM. However this file also contains the following:

### `clean_data(data)`
Used to eliminate NaN/Inf values in features extracted from mouse pose.

##### Input:
* `data` **(2D numpy array)**: a (time x features) array of pose features.

##### Output:
* `data_clean` **(2D numpy array)**: same array, with NaN and Inf values replaced by the last value that was neither.
 
 
 ### `apply_windowing(starter_features)`
 Applies MARS-style temporal windowing to feature values: for each feature, on each frame, computes the mean, standard deviation, min, and max within a window of 3, 11, and 21 frames of the current frame (resulting in a 12-fold increase in feature count).
 ##### Input:
 * `starter_features` **(2D numpy array)**: a (time x features) array of pose features.
 
 ##### Output:
 * `windowed_features` **(2D numpy array)**: now a (time x 12*features) array of windowed pose features.
 
 ### `apply_wavelet_transform(starter_features)`
 A simple script for convolving pose features with a set of wavelets. Frequency range of wavelets (`scales`) is currently hard-coded in the function but this could easily be modified.
 
 ##### Input:
 * `starter_features` **(3D numpy array)**: a (mouse ID x time x features) array of pose features.
 
 ##### Output:
 * `transformed_features` **(2D numpy array)**: a new array, now with dimensions (time x features * (1 + len(`scales`)) * 2 * 2): each feature has been convolved with each of two wavelets that have been scaled by each value in `scales`. Feature values for the two mice are now provided in columns of `transformed_features` ie `transformed_features = [[all convolved mouse 1 features] [all convolved mouse 2 features]]`.



## MARS_annotation_parsers.py
Contains scripts for reading and writing behavior annotations. The relevant function is:

### `parse_annotations(fid, use_channels=[], timestamps=[])`
Reads annotations from a `*.txt` or `*.annot` file and returns them in a dictionary.
##### Inputs:
* `fid` **(string)**: path to an annotation file (`*.txt` or `*.annot`).
* `use_channels` **(string array)**: for `*.annot` file, names of channels to load annotations from (default: merges annotations from all channels).
* `timestamps` **(list)**: for `*.annot` files, exact timestamps of frames in the annotated `*.seq` movie, to ensure accurate conversion from annotated times to frame numbers (if not provided, will perform conversion using framerate specified in the `*.annot` file header).

##### Output:
 A **dict** with the following keys:
* `behs` **(string array)**: list of behaviors annotated for in the file (some may not actually appear).
* `nstrm` **(int)**: number of channels annotated (typically 1 or 2).
* `nFrames` **(int)**: number of frames in the annotated video.
* `behs_frame` **(string array)**: a string array of length `nFrames`. Each entry is a single behavior from `behs`, or "other" if no actions were annotated on that frame. 
* (Plus a few other keys that you can ignore)


