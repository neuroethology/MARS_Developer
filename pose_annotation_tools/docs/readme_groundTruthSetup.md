# Setting up and running annotation jobs on AWS

We will be using SageMaker Ground Truth to send batches of images to a human workforce to be annotated for animal pose. SageMaker supports multiple types of crowdsourced data annotation-- such as text interpretation, bounding box creation, and object identification-- by a public or private workforce. You can read more in-depth documentation in the [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html).

These instructions require an AWS account (with a linked credit card if you are using a paid or public workforce). You can create an account by visiting [console.aws.amazon.com](http://console.aws.amazon.com).

This tutorial will walk you through the following steps:
* [Importing data pre- and post-processing functions](#importing-the-pre--and-post-processing-lambda-functions) (one time only)
* [Preparing your data for annotation](#preparing-your-data-for-annotation)
    * [Uploading images to the cloud](#uploading-images-to-the-cloud)
    * [Creating a data manifest and setting up bucket access](#creating-a-data-manifest-and-setting-up-bucket-access)
* [(Optional) Creating a private annotation workforce](#(optional)-creating-a-private-annotation-workforce) (one time only)
* [Submitting your labeling job](#submitting-your-labeling-job)
    * [Uploading the annotation interface](#editing-the-annotation-interface)
    * [Setting job parameters and running](#setting-job-parameters-and-running)
    * [Downloading the completed annotations](#downloading-the-completed-annotations)


## Importing the pre- and post-processing Lambda functions
Lambda functions are Python scripts that are called when transferring data between Amazon services-- for example when sending your images off to be annotated, and when collecting the annotators' labels. Ground Truth labeling jobs use two Lambdas:

* A pre-processing Lambda that helps to customize input to the annotation interface.
* A post-processing Lambda, an accuracy improvement algorithm to tell Ground Truth how it should assess the quality of labels that humans provide.

These Lambdas are already written, we just have to import them from Amazon's Serverless Application Repository to our own AWS account. This only has to be done once, the very first time you run a labeling job.

#### To import the Lambda functions:
1. In your [AWS Management Console](http://console.aws.amazon.com), search for “Serverless Application Repository” and select it. ([screenshot](serverlessapprepo.png))
2. On the menu on left hand side, click on <kbd>Available applications</kbd>, search for “Ground Truth” and select `aws-sagemaker-ground-truth-recipe`. ([screenshot](groundtruthlambda.png))
3. In the following screen, scroll down to the bottom to find the <kbd>Deploy</kbd> button and click on it.
4.	Deployment of this application may take few minutes. Please wait until you see [this screen](docs/lambdasdeployed.png), showing that 4 AWS resources (2 Lambdas and 2 Roles) were created.

Now you have custom Lambdas, to be used in running labeling jobs, imported successfully in your account.

## Preparing your data for annotation
To collect a new set of MARS training data, you must first extract frames from videos collected in your experimental setup. ([Click here to learn how to perform frame extraction using MARS](../readme.md#extract-video-frames-for-annotation).) Once extracted, you must upload those frames to AWS where they can be used in an annotation job. This includes the following steps:

### Uploading images to the cloud
1. Back on [AWS](http://console.aws.amazon.com), search "S3" in the Find Services menu and go to Amazon S3 (Scalable Storage in the Cloud).
2. Click the <kbd>+ Create bucket</kbd> button to make a new S3 bucket for your images. ([screenshot](s3bucket.png))
3. Set a name for your bucket, and set the region to be the same region in which you imported your Lambda functions. You may leave all other default settings as is, then scroll down to click <kbd>Create bucket</kbd>.
    > Note: by default, no one other than you will be able to access images in your bucket (aside from annotators who are served the images via SageMaker.)

4. Click the name of your new bucket, then select <kbd>Upload</kbd> in the next screen. In the Upload page, add files/folders, then scroll to the bottom and click <kbd>Upload</kbd>.

### Creating a data manifest and setting up bucket access
The data manifest is a .json file that tells Ground Truth which files from your S3 bucket should be annotated. If you have metadata that you would like to keep associated with your images, you can [generate this file yourself](docs/readme_customManifests.md) and upload it to your S3 bucket. Otherwise, SageMaker can crawl an S3 bucket and generate a manifest automatically.

We will now create a manifest for your data (if you don't have one), and while we're there create an element called an "IAM Role" that will allow SageMaker to access the contents of your bucket.

1. Navigate to the SageMaker console by logging into [AWS](http://console.aws.amazon.com) and searching "SageMaker" in the Find Services menu.
2. Choose <kbd>Labeling jobs</kbd> in the left navigation pane, and then click the <kbd>Create labeling job</kbd> button. We're not actually going to create a labeling job right now, but we will use the menu that pops up to generate the manifest and IAM Role.
3. If you already have a manifest file, skip to step 6 of these instructions. Otherwise, scroll down to the "Data Setup" section.
4. Under "S3 location for input datasets", click <kbd>Browse S3</kbd>, select your bucket, then scroll down and click <kbd>Choose</kbd>.
5. Under "Data Type" select "Image". Skip over the IAM Role field (we'll do this next) and click <kbd>Complete data setup</kbd>. After a second of processing, you will find the full path to your new manifest file in the "S3 location for input datasets" field- keep this name for the next section.
6. Next, open the menu under "IAM Role" and select "Create a new role". In the window that pops up, you can set which buckets SageMaker has access to. Under "S3 buckets you specify", either select "Specific S3 buckets" and enter the name of your data bucket, or to make life easier select "Any S3 bucket" (this will let you re-use the same role for all of your labeling jobs.) Then click <kbd>Create</kbd>.
7. That's it for now- you may exit out of this labeling job. In the next step, we'll be retrieving the IAM Role that you just created, and using it (and some other information) to submit your job programmatically.

## (Optional) Creating a private annotation workforce
Ground Truth sends your data to human annotators to label your animal's pose. By default, these annotators will be users of [Amazon Mechanical Turk](https://www.mturk.com/) (MTurk), a crowdsourcing marketplace that sends your task to a global workforce. MTurk annotations can be noisy- we compensate for this in the next step by having 5 workers label each image (a minimum of 3 workers/image is recommended.)

If your data is sensitive or especially difficult to annotate, you may not want to rely on a public workforce. In this case, you can create your own **private annotation team**, allowing you and your colleagues to annotate data within the Ground Truth interface. If desired, follow [these instructions to create a private workforce](docs/readme_privateWorkforce.md).

## Submitting your labeling job
Finally, it's time to make a labeling job. This consists of uploading an annotation interface, and then setting some job parameters within a Jupyter notebook.

### Uploading the annotation interface
The annotation interface is a simple piece of HTML with instructions for annotators. The MARS interfaces also include a script that makes sure each body part is clicked exactly once per image.

We have provided [two example interfaces](../annotation_interface) (`.template` files) for annotating top- and front-view movie frames in a standard home cage. You can modify the `short-instructions` section, the `full-instructions` section, the display images, and the body part label names (line 9). Change the value of `num_object` (line 43) if you want workers to annotate multiple instances of a keypoint in each image. (Note that if your animals can be visually distinguished, it is best to create distinct keypoint names for each animal, eg for the white vs black mouse in MRAS.)

After modifying the interface as you see fit, open [AWS](http://console.aws.amazon.com) and navigate to S3. Create a new S3 bucket, give it a name, and *uncheck* the "Block *all* public access" box so objects in the bucket can be accessed. Then scroll down and click <kbd>Create bucket</kbd>. Upload the interface `.template` along with any instructional images you want to include-- make sure to update your `.template` HTML to include the full URL of these images.

### Setting job parameters and running
At last, it is labeling job time! We will run the job from [this Jupyter notebook](../MARS_annotation_tools_demo.ipynb), after connecting to AWS. You have two options here: running the notebook from within a SageMaker notebook instance, or running on your local machine after installing the AWS command-line interface (CLI).

#### Using a SageMaker notebook instance
1. Navigate to the SageMaker console on [AWS](http://console.aws.amazon.com), and click <kbd>Notebook instances</kbd> in the left-hand menu, then click the <kbd>Create notebook instance</kbd> button.
2. Give the notebook a name, and under "Permissions and Encryption" set the IAM role to `AmazonSageMaker-ExecutionRole-xxxxxxxxxxx`. Under "Git repositories", select `Clone a public Git repository to this notebook instance only`, and add the path to this repository (http://github.com/neuroethology/MARS_developer). Finally, scroll to the bottom and click <kbd>Create notebook instance</kbd>.
3. Your new notebook should now appear in your list of notebook instances. Click <kbd>Start</kbd> under "Actions" for this notebook, and wait a few minutes for the notebook status to update from `Pending` to `InService`, then click <kbd>Open Jupyter</kbd>.
4. Navigate to [../MARS_annotation_tools_demo.ipynb](../MARS_annotation_tools_demo.ipynb) and follow instructions from there!

> **Note:** Don't forget to stop the notebook instance (select the instance and click `Actions`>`Stop` in the Notebook instances menu) once the job has been submitted, so you are not billed for leaving it running!

#### Using AWS CLI
I still have to figure out how to do this!

### Downloading the completed annotations
