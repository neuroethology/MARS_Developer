# POSE ESTIMATION - STACKED HOURGLASS NETWORK

This is an implementation of **Stacked Hourglass Networks for Human Pose Estimation**, proposed by Newell et al. and
adapted by us for mouse pose estimation (for use in the MARS system).

[https://arxiv.org/abs/1603.06937](https://arxiv.org/abs/1603.06937)

All the related files of dataset and model can be found [here](https://www.dropbox.com/home/team_folder/MARS/tf_dataset_keypoints/top_correct)

![alt text](http://www-personal.umich.edu/~alnewell/images/stacked-hg.png)

###  TRAIN AND TEST HOURGLASS POSE DETECTION

##### STEPS:

1. Use files in AMT_annotation folder to recreate the tf format files with the parts included.
2. (Optional) Use `compute_sigmas.py` to compute sigmas (the standard deviation in 'turker click distance --analogous to the spatial ambiguity of a part's location) and save them for use in the config file.
    We can also simply define sigma ourselves --it's simply the standard deviation of the gaussian being generated around each keypoint.
3. Next we need a configuration file: save it in the dataset folder, one file for each type of dateset.  
    Parameters are described in detail below.
4. `visualize_inputs.py` to visualize if the inputs created are correct  
5. `train.py` train the hourglass net  
6. Regularly evaluate the model using `eval.py`, which uses the COCOeval framework to find the Precision and Recall of the current model.
7. `visualize_eval.py` to visualize evaluation of keypoints detectetion with separate images for each keypoint  
8. `visualize_eval_heatmaps.py` visualize heatmaps predictions  
9. `detect.py` detect keypoints on unseen images  
10. `visualize_detect.py` visualize predicted keypoints
11. Once validation performance has sufficiently plateaued, extract the model using `export_pose.py` --this exported model is what's used in MARS.


#### DETAILS:

#### Setup

##### A Note on System Requirements

For training MARS, we used an HPZ640 Workstation with an 8-core Intel Xeon CPU, 24 GB of RAM, and a 12 GB Titan Xp.

That said, training MARS predominantly makes use of the GPU, so training speed is mostly dependent on the quality of your
GPU --we've also trained models using a GTX 960, but that can take ~30-50% more time.


##### COCO evaluation

Note: You can download COCO evaluation tool from [here](https://github.com/pdollar/coco). Remember to run make on PythonAPI in order to get it work

##### Dataset Setup:

Once you have your annotation files, modify `AMT2tfrecords_basic.py`(a script that converts inputs to tfrecord format)
to contain your annotation information. Everything you need to change should have a **TODO** marked next to it in the code.
Once you've modified what you need to, run the script to generate your tfrecord files.

By the end of this, you should have a directory containing a number of tfrecord files --this is what we call **$DATASET_DIR**.




##### Inputs

The data needs to be stored in an example protocol buffer. The protocol buffer will have the following fields:

|Key|	                       Value|
|---------------------|-----------|
|image/id	      |         string containing an identifier for this image.|
|image/filename |            string containing a file system path to the of the image file.|
|image/encoded	 |          string containing JPEG encoded image in RGB colorspace|
|image/height	 |          integer, image height in pixels|
|image/width	 |              integer,  image width in pixels|
|image/colorspace	|       string, specifying the colorspace, e.g. 'RGB'|
|image/channels	 |          integer, specifying the number of channels, e.g. 3|
|image/format	  |         string, specifying the format, e.g. 'JPEG'|
|image/extra	  |             string, any extra data can be stored here. For example, this can be a string encoded json structure.|
|image/class/label|	       integer specifying the index in a classification layer. The label ranges from [0, num_labels), e.g 0-99 if there are 100 classes.|
|image/class/text	|       string specifying the human-readable version of the label |
|image/class/conf	|       float value specifying the confidence of the label. For example, a probability output from a classifier.|
|image/object/count	|       an integer, the number of object annotations. For example, this should match the number of bounding boxes.|
|image/object/area	|       a float array of object areas; normalized coordinates. For example, the simplest case would simply be the area of the bounding boxes. Or it could be the size of the segmentation. Normalized in this case means that the area is divided by the (image width x image height)|
|image/object/id	 |          an array of strings indicating the id of each object.|
|image/object/bbox/xmin	|   a float array, the left edge of the bounding boxes; normalized coordinates.|
|image/object/bbox/xmax	|   a float array, the right edge of the bounding boxes; normalized coordinates.|
|image/object/bbox/ymin	|   a float array, the top left corner of the bounding boxes; normalized coordinates.|
|image/object/bbox/ymax	|   a float array, the top edge of the bounding boxes; normalized coordinates.|
|image/object/bbox/score	|   a float array, the score for the bounding box. For example, the confidence of a detector.|
|image/object/bbox/label|	   an integer array, specifying the index in a classification layer. The label ranges from [0, num_labels)|
|image/object/bbox/text|	   an array of strings, specifying the human readable label for the bounding box.|
|image/object/bbox/conf	|   a float array, the confidence of the label for the bounding box. For example, a probability output from a classifier.|
|image/object/parts/x	 |  a float array of x locations for a part; normalized coordinates.|
|image/object/parts/y	|   a float array of y locations for a part; normalized coordinates.|
|image/object/parts/v	|   an integer array of visibility flags for the parts. 0 indicates the part is not visible (e.g. out of the image plane). 1 indicates the part is occluded. 2 indicates the part is visible.|
|image/object/parts/score |  a float array of scores for the parts. For example, the confidence of a keypoint localizer.|

**Take note:**
* Many of the above fields can be empty. Most of the different systems using the tfrecords will only need a subset of the fields.
* The bounding box coordinates, part coordinates and areas need to be normalized. For the bounding boxes and parts this means that the x values have been divided by the width of the image, and the y values have been divided by the height of the image. This ensures that the pixel location can be recovered on any (aspect-perserved) resized version of the original image. The areas are normalized by they area of the image.
* The origin of an image is the top left. All pixel locations will be interpreted with respect to that origin.

The `create_tfrecords.py` file has a convience function for generating the tfrecord files. You will need to preprocess
your dataset and get it into a python list of dicts. Each dict represents an image and should have a structure that
mimics the tfrecord structure above. However, slashes are replaced by nested dictionaries, and the outermost image
dictionary is implied. Here is an example of a valid dictionary structure for one image:

```python
image_data = {
  "filename" : "/path/to/image_1.jpg",
  "id" : "0",
  "class" : {
    "label" : 1,
    "text" : "Indigo Bunting",
    "conf" : 0.9
  },
  "object" : {
    "count" : 1,
    "area" : [.49],
    "id" : ["1"],
    "bbox" : {
      "xmin" : [0.1],
      "xmax" : [0.8],
      "ymin" : [0.2],
      "ymax" : [0.9],
      "label" : [1],
      "score" : [0.8],
      "conf" : [0.9]
    },
    "parts" : {
      "x" : [0.2, 0.5],
      "y" : [0.3, 0.6],
      "v" : [2, 1],
      "score" : [1.0, 1.0]
    }
  }
}
```

If the `encoded` key is not provided, then the `create` method will read in the image by using the filename value.
In this case, it is assumed that image is stored in either jpg or png format. If `encoded` is provided,
then it is required to provide `height`, `width`, `format`, `colorspace`, and `channels` as well.

Once you have your dataset preprocessed, you can use the create method in create_tfrecords.py to create the tfrecords files. For example:

```python
# this should be your array of image data dictionaries.
# Don't forget that you'll want to separate your training and testing data.
train_dataset = [...]

from create_tfrecords import create
failed_images = create(
  dataset=train_dataset,
  dataset_name="train",
  output_directory="./train_dataset",
  num_shards=10,
  num_threads=5
)
```

This call to the `create` method will use 5 threads to produce 10 tfrecord files, each prefixed with the name train in the
directory `tf_dataset_keypoints`.

All images that cause errors will be returned to the caller. An extra field, `error_msg`, will be added to the dictionary
for that image, and will contain the error message that was thrown when trying to process it.
Typically an error is due to filename fields that don't exist.

```python
print("%d images failed." % (len(failed_images),))
for image_data in failed_images:
  print("Image %s: %s" % (image_data['id'], image_data['error_msg']))
 ```

Next, you'll need to create a configuration file. Checkout the example to see the different settings.
Some especially important settings include:
Next, you'll need to setup your **$EXPERIMENT_DIR**. The only thing you need to do is put some valid config files into
your desired directory. To do this you'll have to create a new (or modify an existing) configuration file. Config files
are formatted using yaml --there are a variety of resources describing the YAML format, so that's omitted here.

**NOTE:** An example configuration file is available with the rest of the codebase.

Configuration parameters include:
  - **Keypoint Parameters**
      * PARTS:
          * NUM_PARTS: the number of parts
          * LEFT\_RIGHT\_PAIRS: pairs of indices corresponding to symmetric parts (e.g. [1,2] for the ears)
          * NAMES: labels of the parts (e.g. ['right_ear','left_ear',...])
          * SIGMAS: Heatmaps are made by taking the keypoint location and convolving it with a 2d gaussian --this list of floats is the standard deviation of the gaussian for each part.
  - **Image Augmentation Parameters**
      * LOOSE\_BBOX\_CROP: when set to true, allows you to expand the bbox by the fraction specified in LOOSE\_BBOX\_PAD\_FACTOR
          * LOOSE\_BBOX\_PAD\_FACTOR: if the above is set to true, expands the bbox by this fraction.
      * DO_RANDOM_PADDING: when set to true, uses a bounding box of variable size to train on.
	  * RANDOM_PADDING_FREQ: The frequency with which to use a random bounding box padding, otherwise we use LOOSE_BBOX_PAD_FACTOR.
	  * RANDOM_PADDING_MIN: The minimum padding percentage possible to randomly draw.
	  * RANDOM_PADDING_MAX: The maximum padding percentage possible to randomly draw.
      * DO_COLOR_DISTORTION: The fraction of the time we should distort the color (by modifying gamma, contrast, or brightness.)
      * DO_RANDOM_BLURRING: when set to true, applies a gaussian blur to the input image.
          * RANDOM_BLUR_FREQ: The frequency with which to apply this blur.
          * MAX_BLUR: The maximum size of the blurring kernel.
      * DO_RANDOM_NOISE: when set to true, applies gaussian white noise to the image.
          * RANDOM_NOISE_FREQ: The frequency with which to apply the above noise.
          * RANDOM_NOISE_SCALE: The standard deviation of the noise generator.
      * DO_RANDOM_ROTATION: When set to true, allows us to rotate the image.
          * RANDOM_ROTATION_FREQ: The frequency with which to rotate the image.
          * RANDOM_ROTATION_DELTA: The maximum change in angular orientation (in degrees).
      * DO_JPEG_ARTIFACTS: When set to true, allows the application of jpeg degradation to images.
          * RANDOM_JPEG_FREQ: Frequency with which to perform jpeg degradation.
          * RANDOM_JPEG_QUALITY_MIN: The minimum quality of the jpeg encoding.
          * RANDOM_JPEG_QUALITY_MAX: The maximum quality of the jpeg encoding.
      * DO_RANDOM_BBOX_SHIFT:  Ther frequency with which to perturb the coordinates of the bounding box --As in (1), we set this to 0.
      * DO_RANDOM_FLIP_LEFT_RIGHT: The frequency with which to use horizontally-flipped images. This should help with model generalization, assuming that our model is left-right independent (and it should be).
  - **Input Queue Parameters**
      * NUM_INPUT_THREADS:  The number of threads to use, when constructing input examples. Depends on your hardware setup --generally you want as many threads as processor cores.
      * BATCH_SIZE: The number of images to process in one iteration. Depending on your hardware setup, you will need to adjust this parameter so that everything fits in memory.
      * NUM_TRAIN_EXAMPLES: The number of training examples in your dataset. This, along with BATCH_SIZE, is used to compute the number of iterations in an epoch (an epoch has passed when all examples have been seen once).
      * NUM_TRAIN_ITERATIONS: The maximum number of training iterations to perform before stopping.
  - **Learning Parameters**
      * LEARNING_RATE_DECAY_TYPE: Choose between *fixed*,*exponential*, and *polynomial*. At the moment, only fixed is implemented.
      * INITIAL_LEARNING_RATE: As in (1), we use 0.00025.

Once you've set up your experiment and dataset directories, you can begin training the pose model.

|Key	      |                     Description|
|--------------------------------|-----------|
|NUM_PARTS     |                    number of parts|
|LEFT_RIGHT_PAIRS |                 index of the parts that belong to symmetric parts|
|NAMES         |                    labels of the parts|
|SIGMAS                     |       the array you computed from the previous step|
|LOOSE_BBOX_CROP               |    recommended to set as false. It allows to pad the box by a factor|
|DO_RANDOM_FLIP_LEFT_RIGHT   |      percentage of time you want to use flipped images|
|DO_RANDOM_BBOX_SHIFT   |           perturbation of the coordinates of the bounding box|
|DO_COLOR_DISTORTION   |            fraction of the time to distort color|
|NUM_INPUT_THREADS      |           Depending on your hardware setup, the number of threads to use|
|BATCH_SIZE             |           Depending on your hardware setup, you will need to adjust this parameter so that the network fits in memory. The number of images to process in one iteration.|
|NUM_TRAIN_EXAMPLES	|              This, along with BATCH_SIZE, is used to compute the number of iterations in an epoch. It's the number of images in your training tfrecords|
|NUM_TRAIN_ITERATIONS     |         This is how many iterations to run before stopping.|
|LEARNING_RATE_DECAY_TYPE |         chose betwettn "fixed',"exponential", "polynomial", Fixed is recommened|
|INITIAL_LEARNING_RATE    |         you want to set is a bit low, 0.00025|
|NUM_EPOCHS_PER_DELAY   |           very important parameter. If you don't early stop and check how is going the training, this parameter anyway lower ther learning rate after the set epoch. You don't want to lower it too early|
|LEARNING_RATE_DECAY_FACTOR |       factor to decrease the learning rate|

You'll definitely want to go through the other configuration parameters, but make sure you have the above parameters set correctly.

Now that you have your dataset in tfrecords, and you set up your configuration file, you'll be able to train the pose model.
First, you can debug your image augmentation setting by visualizing the inputs to the network:

```sh
python visualize_inputs.py \
--tfrecords $DATASET_DIR/train* \
--config $EXPERIMENT_DIR/config_train.yaml
```
When you are ready you can start training the model. This time, compared to MULTIBOX we train from scratch
```sh
python train.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/logdir \
--config $EXPERIMENT_DIR/config_train.yaml
```

If you want to personalize more the parameters of the train regardless the configuration file you can use

```sh
python train_custom.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/ \
--config $DATASET_DIR/config_train.yaml \
--max_number_of_steps 20000 \
--lr 0.00005 \
--lr_decay_type 'fixed' \
--batch_size 4
```
If you want to evalaute the prediction on the validadation set you can run 
```sh
python eval.py \
--tfrecords $DATASET_DIR/test*
--checkpoint_path $EXPERIMENT_DIR/
--summar_dir $EXPERIMENT_DIR/
--config $DATASET_DIR/config_test.yaml \
--max_iterations 200


If you have a validation set, you can visualize the ground truth boxes and the predicted boxes:
```sh
 python visualize_eval.py \
--tfrecords $DATASET_DIR/test* \
--checkpoint_path $EXPERIMENT_DIR/ \
--config $EXPERIMENT_DIR/config_detect.yaml
```
At "application time" you can run the detect script to generate predicted keypoints on new images.
You can debug your detection setting by using another visualization script:

**Detect parts on new data**
```sh
python detect.py \
--tfrecords $DATASET_DIR/test* \
--checkpoint_path $EXPERIMENT_DIR/logdir \
--config $EXPERIMENT_DIR/config_detect.yaml \
--save_dir $EXPERIMENT_DIR/logdir/detections
```
**Visualize detections on new data**
```sh
python visualize_detect.py \
--tfrecords $DATASET_DIR/test* \
--checkpoint_path $EXPERIMENT_DIR/ \
--config $EXPERIMENT_DIR/config_detect.yaml
```
#### Export:

Once the model is finished training, we export it using:
```sh
python export_pose.py \
--checkpoint_path $EXPERIMENT_DIR/ \
--config $EXPERIMENT_DIR/config_test.yaml
--export_dir $EXPERIMENT_DIR/ \
--savename my_finished_model
```

####  FILES INCLUDED IN THIS FOLDER

|Name|                             Description|
|----------------------------------|-----------|
|csv_extract_parts_actions_top.py |  extract from csv the annotations and parts and creates a dictiorary|
|AMT2tfrecords.py               |    convert dictionary data to tf format and split it into train, val, test dataset|
|AMT2tfrecords_separate.py      |    convert dictiornary data of one mouse to tf format and split it into train, val, test dataset|
|compute_sigmas.py     |             compute sigmas from annotation to insert in the configuration file|
|config.py               |           read configuration file|
|create_tfrecords.py      |          utility to convert dictionary to tf format|
|heatmaps_tfrecords.py       |       if you want to precompute heatmaps|
|inputs.py                   |       utility to prepare input pipeline to network|
|train_inputs.py               |     input nodes to pipeline|
|train_inputs_precomputed.py    |    this is used when you precomputed the heatmaps using heatmaps_tfrecords|
|train_inputs_old.py            |    ignore this|
|visualize_inputs.py           |     visualize that the tf records created are ok as well as the heatmaps|
|model.py                      |     hourglass model|
|model_old.py                  |     ignore this|
|loss.py                        |    loss function|
|train.py                        |   train|
|train_custom.py                 |   train with possibility to set on the fly parameters of learning and optimization regardless the config file|
|val_keypoints.sh                 | utility to evaluate training with validation set to see when to stop the training stage|
|eval.py                          |  utlity to evalaute hourglass with coco eval saving pickle file for other utilities|
|eval_original.py                 |  same as eval but used with matteo_exp_coco to save images|
|eval_input.py                   |   input pipeline for eval|
|visualize_eval.py               |   visualize evaluation of keypoints detection with separate images for each keypoint|
|visualize_eval_heatmaps.py      |   visusalize evaluation of keypoints heatmaps|
|detect.py                       |   inference of keypoints on unseen images|
|detect_inputs.py                |   utility to  prepare input batch for detection|
|detect_inputs_imsize.py         |   utility to  prepare input batch for detection  using image size|
|visualize_detect.py              |  visualize predicted keypoints on different images per keypoint|
|detect_wbbox.py                  |  optimized version of detect where we impose to find the maxima within the bounding box rescaled to input size and image size coordinates of keypoints|
|my_plot_pose_conf.py             |  plot predicted keypoints taking into accounts cofindence and drawing skeleton|
|my_ply_pose_congs_separate.py    |  same but for separate mice detection|
|my_prcurve_prep_data.py           | similar to eval saves gt and prediction for later use|
|my_prcurve.py                     | compute pr curves from save evaluation|

