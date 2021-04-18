# Setting up and running annotation jobs on AWS

We will be using SageMaker Ground Truth to send batches of images to a human workforce to be annotated for animal pose. SageMaker supports multiple types of crowdsourced data annotation-- such as text interpretation, bounding box creation, and object identification-- by a public or private workforce. You can read more in-depth documentation in the [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html).

These instructions require an AWS account (with a linked credit card if you are using a paid or public workforce). You can create an account by visiting [console.aws.amazon.com](http://console.aws.amazon.com).

This tutorial will walk you through the following steps:
1. [Import data pre- and post-processing functions](#1-import-the-pre--and-post-processing-lambda-functions) **(one time only)**
2. [Prepare your data for annotation](#2-prepare-your-data-for-annotation)
    * [Create S3 buckets for your data and annotation interface](#create-s3-buckets-for-your-data-and-annotation-interface)
    * [Prepare your annotation interface](#prepare-your-annotation-interface)
    * [Upload files](#upload-files)
    * [Set up bucket access](#set-up-bucket-access) **(one time only)**
3. [(Optional) Create a private annotation workforce](#3-optional-create-a-private-annotation-workforce) **(one time only)**
4. [Submit your labeling job](#4-submit-your-labeling-job)
5. [Download the completed annotations](#5-download-the-completed-annotations)


## 1. Import the pre- and post-processing Lambda functions
Lambda functions are Python scripts that are called when transferring data between Amazon services-- for example when sending your images off to be annotated, and when collecting the annotators' labels. Ground Truth labeling jobs use two Lambdas:

* A pre-processing Lambda that helps to customize input to the annotation interface.
* A post-processing Lambda, an accuracy improvement algorithm to tell Ground Truth how it should assess the quality of labels that humans provide.

These Lambdas are already written, we just have to import them from Amazon's Serverless Application Repository to our own AWS account. This only has to be done once, the very first time you run a labeling job.

#### To import the Lambda functions:
1. In your [AWS Management Console](http://console.aws.amazon.com), search for “Serverless Application Repository” and select it. ([screenshot](serverlessapprepo.png))
2. Double-check **region** you are operating in-- in the upper-right corner of your AWS console, next to <kbd>Support</kbd>, you should see a location name. Click on this to see a list of available regions, as well as hyphenated **region names**. For example, the region "US East (Ohio)" has region name **us-east-2**. Make a note of this name for later.
3. On the menu on left hand side, click on <kbd>Available applications</kbd>, search for “Ground Truth” and select `aws-sagemaker-ground-truth-recipe`. ([screenshot](groundtruthlambda.png))
4. In the following screen, scroll down to the bottom to find the <kbd>Deploy</kbd> button and click on it.
5.	Deployment of this application may take few minutes. Please wait until you see [this screen](lambdasdeployed.png), showing that 4 AWS resources (2 Lambdas and 2 Roles) were created.

Now you have custom Lambdas, to be used in running labeling jobs, imported successfully in your account.

## 2. Prepare your data for annotation
(If you haven't yet created a dataset to annotate, [follow these steps to extract frames from videos](../readme.md#extract-video-frames-for-annotation).)

In this step, we'll upload two sets of files to Amazon:
* the images you would like workers to annotate
* the html template for an annotation interface that will tell workers which body parts to label

### Create S3 buckets for your data and annotation interface

First, we're going to make two "buckets" on Amazon: one will host the **data to be annotated**, and the other will host an **annotation interface** with instructions for your annotators.

For the data:
1. Back on [AWS](http://console.aws.amazon.com), search "S3" in the Find Services menu and go to Amazon S3 (Scalable Storage in the Cloud).
2. Click the <kbd>+ Create bucket</kbd> button to make a new S3 bucket for your images. We'll be calling this `data_bucket`. ([screenshot](s3bucket.png))
3. Set a name for your bucket, and set the region to be **the same region in which you imported your Lambda functions**. You may leave all other default settings as is, then scroll down to click <kbd>Create bucket</kbd>.
    > Note: by default, no one other than you will be able to access images in your bucket (aside from annotators who are served the images via SageMaker.)

For the annotation interface:
1. Create a new S3 bucket for the annotation interface, give it a name, and *uncheck* the "Block *all* public access" box so objects in the bucket can be accessed. Then scroll down and click <kbd>Create bucket</kbd>. We'll be calling this `template_bucket`.

### Prepare your annotation interface
Now we're going to create the **annotation interface**, a simple piece of HTML with instructions for annotators.

We will create this programmatically from a configuration file called `annot_config.yml` that was created inside `project_name/annotation_data` when you [initialized your labeling project](../pose_annotation_tools#0-initialize-a-new-labeling-project).

1. Open `annot_config.yml` in your text editor. If you are using the same setup as MARS, you can leave this as is, EXCEPT change the names of `data_bucket`, `template_bucket`, and `region` to reflect where you created your S3 buckets, and update the image path under "full_instructions" to include `region` and `template_bucket`.
2. From the terminal, call `python generate_AWS_template.py '/fullpath/to/project_name/annotation_data/annot_config.yml'`

This will create a file `project_name/annotation_data/annotation_interface.template` file containing your annotation template.

### Upload files
Now we're going to upload everything to your two S3 buckets.
1. Open [AWS](http://console.aws.amazon.com) and navigate to S3. First, click on the name of your `data_bucket`, then select <kbd>Upload</kbd> in the next screen. 
2. Upload the images in `project_name/annotation_data/raw_images` (the images you'd like annotated).
3. Next, open your `template_bucket` and upload the interface `.template` along with any instructional images you want to include. If you are using the MARS keypoints, you can use [the instructional image provided in this repository](../annotation_interface/front_view_interface/instruction_image_bodyparts.png).

### Set up bucket access
If you haven't run a labeling job before, we need to create an element called an "IAM Role" that will allow SageMaker to access S3.

1. Navigate to the SageMaker console by logging into [AWS](http://console.aws.amazon.com) and searching "SageMaker" in the Find Services menu.
2. Choose <kbd>Labeling jobs</kbd> in the left navigation pane, and then click the <kbd>Create labeling job</kbd> button. We're not actually going to create a labeling job right now, but we will use the menu that pops up to generate the IAM Role.
3. Open the menu under "IAM Role" and select "Create a new role". In the window that pops up, you can set which buckets SageMaker has access to. Under "S3 buckets you specify", either select "Specific S3 buckets" and enter the name of your data bucket, or to make life easier select "Any S3 bucket" (this will let you re-use the same role for all of your labeling jobs.) Then click <kbd>Create</kbd>.
4. You may now exit out of this labeling job. In step 4 we'll be retrieving the IAM Role that you just created, and using it (and some other information) to submit your job programmatically.

## 3. (Optional) Create a private annotation workforce
Ground Truth sends your data to human annotators to label your animal's pose. By default, these annotators will be users of [Amazon Mechanical Turk](https://www.mturk.com/) (MTurk), a crowdsourcing marketplace that sends your task to a global workforce. MTurk annotations can be noisy- we compensate for this in the next step by having 5 workers label each image (a minimum of 3 workers/image is recommended.)

If your data is sensitive or especially difficult to annotate, you may not want to rely on a public workforce. In this case, you can create your own **private annotation team**, allowing you and your colleagues to annotate data within the Ground Truth interface. If desired, follow [these instructions to create a private workforce](readme_privateWorkforce.md).

## 4. Submit your labeling job
Finally, it's time to make a labeling job- we'll do this from within a Jupyter notebook hosted on SageMaker.

1. Navigate to the SageMaker console on [AWS](http://console.aws.amazon.com), and click <kbd>Notebook</kbd>-><kbd>Notebook instances</kbd> in the left-hand menu, then click the <kbd>Create notebook instance</kbd> button.
2. Give the notebook a name, and under "Permissions and Encryption" set the IAM role to `AmazonSageMaker-ExecutionRole-xxxxxxxxxxx`. Under "Git repositories", select `Clone a public Git repository to this notebook instance only`, and add the path to this repository (http://github.com/neuroethology/MARS_developer). Finally, scroll to the bottom and click <kbd>Create notebook instance</kbd>.
3. Your new notebook should now appear in your list of notebook instances. Click <kbd>Start</kbd> under "Actions" for this notebook, and wait a few minutes for the notebook status to update from `Pending` to `InService`, then click <kbd>Open Jupyter</kbd>.
4. Navigate to [pose_annotation_tools/Submit_Labeling_Job.ipynb](../Submit_Labeling_Job.ipynb) from within the notebook session, and follow all instructions.
5. Once you have run all cells of the notebook, your job should be submitted. In the SageMaker console, click <kbd>Ground Truth</kbd>-></kbd>Labeling Jobs</kbd>, and you should see a job with name **[BUCKET NAME]-xxxxxxxxxx** in progress.

> **Note:** once the job is running, you should stop your Notebook instance (select the instance and click `Actions`>`Stop` in the Notebook instances menu), so you are not billed for leaving it running!

## 5. Download the completed annotations
When a job has finished running, its status in the Labeling Jobs menu will change to ✔️<span style="color:green">Complete</span>. Once this happens, download your completed annotations by following these steps:

1. Navigate to the S3 console on [aws](http://console.aws.amazon.com). If your images to be annotated were stored in a bucket called "BUCKET_NAME", look for the bucket **[BUCKET NAME]-output** and open it.
2. Inside **[BUCKET_NAME]-output**, open **[BUCKET_NAME]-xxxxxxxxxx** (the name of your most recent labeling job- if you have more than one, pick the one with the highest number), then open **manifests** followed by **output**. You should find a file called **output.manifest**, which contains the raw output of your labeling job.
3. Check the box next to **output.manifest**, and select <kbd>Actions</kbd>-><kbd>Download</kbd>.

And you've made it! Take a deep breath, then return to the [MARS Pose Annotation Tools ReadMe](https://github.com/neuroethology/MARS_Developer/tree/develop/pose_annotation_tools#mars-pose-annotation-tools-a-module-for-crowdsourcing-pose-estimation) and refer to step 3 to visualize the annotations you just collected.
