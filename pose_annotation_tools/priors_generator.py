from pose_annotation_tools.priors import *
import pickle
import tensorflow as tf

# tf.compat.v1.enable_eager_execution()


def _parse_function(example_proto):
    features = {
        'image/id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/count': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example_proto, features)


def convert(tf_r):
    raw_dataset = tf.data.TFRecordDataset(tf_r)
    parsed_dataset = raw_dataset.map(_parse_function)
    imgs = []
    img = {'object': {'bbox': {}}}
    for parsed_record in parsed_dataset.take(len(list(tf.python_io.tf_record_iterator(tf_r[0])))):
        for k in list(parsed_record.keys()):
            list_ver = k.split('/')
            if len(list_ver) == 4:
                img['object']['bbox'][list_ver[-1]] = parsed_record[k]
            if len(list_ver) == 2:
                img[list_ver[-1]] = parsed_record[k]
        imgs.append(img)
    return imgs


def generate_priors_from_data(dataset):
    # Generate the apspect ratios from the training data
    dataset_name = dataset
    if not dataset.endswith(".pkl"):
        dataset = convert([dataset])
    else:
        with open(dataset, 'rb') as fp:
            dataset = pickle.load(fp, encoding='latin1')
    aspect_ratios = generate_aspect_ratios(dataset, num_aspect_ratios=11, visualize=False, warp_bboxes=True)
    p = generate_priors(aspect_ratios, min_scale=0.1, max_scale=0.95, restrict_to_image_bounds=True)

    with open(dataset_name.split('/')[-1] + '_priors.pkl', 'wb') as f:
        pickle.dump(p, f)
    return p
