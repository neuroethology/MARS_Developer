# Multibox

This is an implementation of the Multibox detection system proposed by Szegedy et al. in [Scalable High Quality Object Detection](https://arxiv.org/abs/1412.1441). Currently this repository uses the [Inception-Reset-v2](https://arxiv.org/abs/1602.07261) network as the base network. The post classification network is not currently incorporated in this repository, but any classification network can be used. 

This repo supports Python 2.7. Checkout the [requirements file](requirements.txt) to make sure you have the necessary packages. [TensorFlow r0.11](https://www.tensorflow.org/versions/r0.11/get_started/index.html) is *required*. 
 

The input functions to the model require a specific dataset format. You can create the dataset using the utility functions found [here](https://github.com/gvanhorn38/inception/tree/master/inputs). You'll also need to genertate the priors for the bounding boxes. In the [priors.py](priors.py) file you will find convenience functions for generating the priors. For example, assuming you are in a python terminal (in the project directory):

```python
import cPickle as pickle
import priors

aspect_ratios = [1, 2, 3, 1./2, 1./3]
p = priors.generate_priors(aspect_ratios, min_scale=0.1, max_scale=0.95, restrict_to_image_bounds=True)
with open('priors.pkl', 'w') as f:
  pickle.dump(p, f)
``` 

Instead of hand defining the aspect ratios, you can use your training dataset to cluster the aspect ratios of the bounding boxes. There is a utility function to do this in the [priors.py](priors.py) file. 

Next, you'll need to create a configuration file. Checkout the [example](config.yaml.example) to see the different settings. Some especially important settings include:

| Key | Value|
|-----|------|
| NUM_BBOXES_PER_CELL | This needs to be set the number of aspect ratios you used to generate the priors. The output layers of the model depend on this number. |
| MAX_NUM_BBOXES | In order to properly pad all the image annotation data in a batch, the maximum number of boxes in a single image must be known. |
| BATCH_SIZE | Depending on your hardware setup, you will need to adjust this parameter so that the network fits in memory. |
| NUM_TRAIN_EXAMPLES | This, along with `BATCH_SIZE`, is used to compute the number of iterations in an epoch. |
| NUM_TRAIN_ITERATIONS | This is how many iterations to run before stopping. |

You'll definitely want to go through the other configuration parameters, but make sure you have the above parameters set correctly.

Now that you have your dataset in tfrecords, you generated priors, and you set up your configuration file, you'll be able to train the detection model. First, you can debug your image augmentation setting by visualizing the inputs to the network:

```sh
python visualize_inputs.py \
--tfrecords /Volumes/Untitled/tensorflow_datasets/coco_people/kps/val2000/* \
--config /Volumes/Untitled/models/coco_person_detection/9/config_train.yaml
```

Once you are ready for training, you should download the [pretrained inception-resnet-v2 network](https://research.googleblog.com/2016/08/improving-inception-and-image.html) and use it as a starting point. Then you can run the training script:

```sh
python train.py \
--tfrecords /Volumes/Untitled/tensorflow_datasets/coco_people/kps/val2000/* \
--priors /Volumes/Untitled/models/coco_person_detection/9/coco_person_priors_7.pkl \
--logdir /Users/GVH/Desktop/multibox_train/ \
--config /Users/GVH/Desktop/multibox_train/config_train.yaml \
--pretrained_model /Users/GVH/Desktop/Inception_Models/inception-resnet-v2/inception_resnet_v2_2016_08_30.ckpt
```

If you have a validation set, you can visualize the ground truth boxes and the predicted boxes:

```sh
python visualize_val.py \
--tfrecords /Volumes/Untitled/tensorflow_datasets/coco_people/kps/val2000/* \
--priors  /Volumes/Untitled/models/coco_person_detection/9/coco_person_priors_7.pkl \
--checkpoint_path /Volumes/Untitled/models/coco_person_detection/9/model.ckpt-300000 \
--config /Users/GVH/Desktop/multibox_train/config_train.yaml
```

At "application time" you can run the detect script to generate predicted boxes on new images. You can debug your detection setting by using another visualization script:

```sh
python visualize_detect.py \
--tfrecords /Volumes/Untitled/tensorflow_datasets/coco_people/kps/val2000/* \
--priors /Volumes/Untitled/models/coco_person_detection/9/coco_person_priors_7.pkl \
--checkpoint_path /Volumes/Untitled/models/coco_person_detection/9/model.ckpt-300000 \
--config /Users/GVH/Desktop/multibox_train/config_detect.yaml
```

```sh
python detect.py \
--tfrecords /Volumes/Untitled/tensorflow_datasets/coco_people/kps/val2000/* \
--priors /Volumes/Untitled/models/coco_person_detection/9/coco_person_priors_7.pkl \
--checkpoint_path /Volumes/Untitled/models/coco_person_detection/9/model.ckpt-300000 \
--save_dir /Users/GVH/Desktop/multibox_train/ \
--config /Users/GVH/Desktop/multibox_train/config_detect.yaml \
--max_iterations 20
```

