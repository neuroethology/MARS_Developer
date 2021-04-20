# Initial AWS account setup

 SageMaker supports multiple types of crowdsourced data annotation—such as text interpretation, bounding box creation, and object identification—by a public or private workforce. You can read more in-depth documentation in the [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html).

To use SageMaker, you first need to **create an AWS account**, with a linked credit card if you are using a paid or public workforce. You can create an account by visiting [console.aws.amazon.com](http://console.aws.amazon.com).

Next, we will need to set up a couple data handling functions, and manage some permissions. AWS is composed of many separate services (compute, storage, annotations, etc), and each service needs to be given permission to access files belonging to other services. So when you first create your account, you'll need to set up these inter-service permissions, typically called **IAM Roles** or **Trust Relationships**.

There are three core steps we'll need to perform:
1. [Import data pre- and post-processing functions](#1-import-the-pre--and-post-processing-lambda-functions)
2. [Set up S3 (storage) access](#set-up-s3-storage-access)
3. [(Optional) Create a private annotation workforce](#3-optional-create-a-private-annotation-workforce)

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

## 2. Set up S3 (storage) access
Data on AWS is stored in a service called S3, in objects called "buckets". For GroundTruth to be able to access the bucket that will contain your images, we need to create an IAM Role that will allow SageMaker to talk to S3.

1. Navigate to the SageMaker console by logging into [AWS](http://console.aws.amazon.com) and searching "SageMaker" in the Find Services menu.
2. Choose <kbd>Labeling jobs</kbd> in the left navigation pane, and then click the <kbd>Create labeling job</kbd> button. We're not actually going to create a labeling job right now, but we will use the menu that pops up to generate the IAM Role.
3. Open the menu under "IAM Role" and select "Create a new role". In the window that pops up, you can set which buckets SageMaker has access to. Under "S3 buckets you specify", either select "Specific S3 buckets" and enter the name of your data bucket, or to make life easier select "Any S3 bucket" (this will let you re-use the same role for all of your labeling jobs.) Then click <kbd>Create</kbd>.
4. You may now exit out of this labeling job. In step 4 we'll be retrieving the IAM Role that you just created, and using it (and some other information) to submit your job programmatically.


## 3. (Optional) Create a private annotation workforce
Ground Truth sends your data to human annotators to label your animal's pose. By default, these annotators will be users of [Amazon Mechanical Turk](https://www.mturk.com/) (MTurk), a crowdsourcing marketplace that sends your task to a global workforce. MTurk annotations can be noisy- we compensate for this in the next step by having 5 workers label each image (a minimum of 3 workers/image is recommended.)

If your data is sensitive or especially difficult to annotate, you may not want to rely on a public workforce. In this case, you can create your own **private annotation team**, allowing you and your colleagues to annotate data within the Ground Truth interface. If desired, follow [these instructions to create a private workforce](readme_privateWorkforce.md).
