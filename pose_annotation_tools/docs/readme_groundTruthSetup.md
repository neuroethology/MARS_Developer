# Setting up and running annotation jobs on AWS

We will be using SageMaker Ground Truth to send batches of images to a public workforce to be annotated for animal pose.

These instructions require an AWS account (with a linked credit card if you are using a paid or public workforce). You can create an account by visiting [console.aws.amazon.com](http://console.aws.amazon.com).

This tutorial will walk you through the following steps:
* [Importing data pre- and post-processing functions](#importing-the-pre--and-post-processing-lambda-functions) (one time only)
* [Preparing your data for annotation](#preparing-your-data-for-annotation)
    * [Uploading frames to the cloud](#uploading-images-to-the-cloud)
    * [Creating a data manifest](#creating-a-data-manifest)
* [Submitting your labeling job](#submitting-your-labeling-job)
    * [Setting up SageMaker permissions](#setting-up-sagesaker-permissions)
    * [Setting job parameters and running](#setting-job-parameters-and-running)


## Importing the pre- and post-processing Lambda functions
Lambda functions are Python scripts that are called when transferring data between Amazon services-- for example when sending your images off to be annotated, and when collecting the annotators' labels. Ground Truth labeling jobs use two Lambdas:

* A pre-processing Lambda that helps to customize input to the annotation interface.
* A post-processing Lambda, an accuracy improvement algorithm to tell Ground Truth how it should assess the quality of labels that humans provide.

Lucky for us, the Lambdas we need for our labeling job are already written, we just have to import them from Amazon's Serverless Application Repository to our own AWS account. This only has to be done once, the very first time you run a labeling job.

#### To import the Lambda functions:
1. In your [AWS Management Console](http://console.aws.amazon.com), search for “Serverless Application Repository” and select it. ([screenshot](serverlessapprepo.png))
2. On the menu on left hand side, click on <kbd>Available applications</kbd>, search for “Ground Truth” and select `aws-sagemaker-ground-truth-recipe`. ([screenshot](groundtruthlambda.png))
3. In the following screen, scroll down to the bottom to find the <kbd>Deploy</kbd> button and click on it.
4.	Deployment of this application may take few minutes. Please wait until you see [this screen](docs/lambdasdeployed.png), showing that 4 AWS resources (2 Lambdas and 2 Roles) were created.

Now you have custom Lambdas, to be used in running labeling jobs, imported successfully in your account.

## Preparing your data for annotation
To collect a new set of MARS training data, you must first extract frames from videos collected in your experimental setup. ([Click here to learn how to perform frame extraction using MARS](readme.md#extracting-video-frames-for-annotation).) Once extracted, you must upload those frames to AWS where they can be used in an annotation job. This includes the following steps:

### Uploading images to the cloud
1. Back on [AWS](http://console.aws.amazon.com), search "S3" in the Find Services menu and go to Amazon S3 (Scalable Storage in the Cloud).
2. Click the <kbd>+ Create bucket</kbd> button to make a new S3 bucket for your images. ([screenshot](s3bucket.png))
3. Set a name for your bucket, and set the region to be the same region in which you imported your Lambda functions. You may leave all other default settings as is, then scroll down to click <kbd>Create bucket</kbd>.
    > Note: by default, no one other than you will be able to access images in your bucket (aside from annotators who are served the images via SageMaker.)

4. Click the name of your new bucket, then select <kbd>Upload</kbd> in the next screen. In the Upload page, add files/folders, then scroll to the bottom and click <kbd>Upload</kbd>.

### Creating a data manifest
The data manifest is a .json file that tells Ground Truth which files from your S3 bucket should be annotated. If you have metadata that you would like to keep associated with your images, you can [generate this file yourself](docs/readme_customManifests.md). Otherwise, Amazon SageMaker has a tool that will crawl an S3 bucket and generate a manifest automatically.

#### Creating the manifest automatically with SageMaker
Given an input bucket, SageMaker crawls all of the image files (with extensions .jpg, .jpeg, .png) in that bucket, and creates a manifest with each line as {“source-ref”:”<s3-location-of-crawled-image>”}. To use the SageMaker crawler:
1. Navigate to the SageMaker console by logging into [AWS](http://console.aws.amazon.com) and searching "SageMaker" in the Find Services menu.
2. Choose <kbd>Labeling jobs</kbd> in the left navigation pane, and then click the <kbd>Create labeling job</kbd> button.
3. In the next screen, click <kbd>Create manifest file</kbd>.
4. You may then exit out of this labeling job- we'll be submitting our job programmatically in the next step as it gives us more control of the job creation process.

## Submitting your labeling job

### Setting up SageMaker permissions

### Setting job parameters and running
