# Multibox

This Multibox detection system has been adopted from Grant Van Horn's [Multibox](https://github.com/gvanhorn38/multibox) repository, which itself is an implementation of the Multibox detection system proposed by Szegedy et al. in [Scalable High Quality Object Detection](https://arxiv.org/abs/1412.1441). Currently this repository uses the [Inception-Reset-v2](https://arxiv.org/abs/1602.07261) network as the base network.

**Modifications from the [source repo](https://github.com/gvanhorn38/multibox):**
- Code updated from Python 2 to Python 3
- Now runs in Tensorflow 1.15-gpu
 


###  Associated files
**The documentation below is intended for developers- to train the detector, please refer to `MARS_tutorial.ipynb` in the parent directory.**

#### tfrecords
Ahead of training, the raw training data gets packaged into a tfrecord file; this is performed by scripts within `pose_annotation_tools/annotation_postprocessing.py`. The tfrecord has the following fields:

[tbd]

#### Priors
The Multibox detector is initialized using a set of priors over possible bounding box locations. These are `.pkl` files created by scripts within `pose_annotation_tools/annotation_postprocessing.py` along with the tfrecords.

#### Config files
(Documentation reproduced from [source repo](https://github.com/gvanhorn38/multibox).)

Config files contain model hyperparameters that are used during training. You do not need to create these yourself- they are created in your project directory when it is initialized.

Configuration parameters include:
  - **Input Queue Parameters**
    * NUM_BBOXES_PER_CELL: This needs to be set as the number of aspect ratios you used to generate the priors. The output layers of the model depend on this number.
    * MAX_NUM_BBOXES: In order to properly pad all the image annotation data in a batch, the maximum number of boxes in a single image must be known.
    * BATCH_SIZE: Depending on your hardware setup, you will need to adjust this parameter so that the network fits in memory.
    * NUM_TRAIN_EXAMPLES: This, along with `BATCH_SIZE`, is used to compute the number of iterations in an epoch.
    * NUM_TRAIN_ITERATIONS: This is how many iterations to run before stopping.

You'll definitely want to go through the other configuration parameters, but make sure you have the above parameters set correctly.

