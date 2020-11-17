## Pose annotation tools
This folder contains all the tools needed to collect and process manual annotations of animal poses from a public workforce on Amazon SageMaker Ground Truth. This involves the following steps:

1. **Extract** video frames that you would like to annotate.
2. **Upload** those frames to AWS and design a pose labeling job.
3. **Run** your labeling job to collect body part annotations from a human workforce.
4. **Visualize** the annotations from a completed labeling job.
5. **Compile** completed pose annotations so they are ready to use to train MARS.

If you have already collected annotation data from Ground Truth, or with the DeepLabCut annotation interface, you can skip straight to step 5 to compile those annotations for training MARS.
