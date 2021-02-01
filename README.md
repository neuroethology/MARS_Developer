This repository includes code to:

1) Collect crowdsourced pose annotation data with `pose_annotation_tools`
2) Train or fine-tune the MARS mouse detector with `multibox_detection`.
3) Train or fine-tune the MARS mouse pose estimator with `hourglass_pose`.
4) Evaluate pose estimation performance using the COCO framework in `MARS_pycocotools`.
5) Train new MARS behavior classifiers (code coming soon).

More readme to come!

### Setup

All code should be run within the MARS_Developer conda environment, which is built by calling
```
conda env create -f mars_dev.yml
```

You can run jupyter notebooks from within the MARS_dev environment by installing ipykernel:
```
conda install -c anaconda ipykernel
```
and then calling
```
python -m ipykernel install --user --name=mars_dev
```
