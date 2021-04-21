# Running annotation jobs on AWS

We will be using SageMaker Ground Truth to send batches of images to a human workforce to be annotated for animal pose.

We've divided these instructions into two sets: the one-time-only steps you will need to follow to **set up your account**, and the steps you'll follow any time you **submit a new labeling job**. The initial setup process takes some time investment, but submitting new labeling jobs is straightforward.

**To set up your AWS account for the first time, [follow these instructions]().**

Once you've made it through those instructions, submitting a job with SageMaker is straightforward:

1. [Prepare your data for annotation](#2-prepare-your-data-for-annotation)
    * [Create S3 buckets for your data and annotation interface](#create-s3-buckets-for-your-data-and-annotation-interface)
    * [Prepare your annotation interface](#prepare-your-annotation-interface)
    * [Upload files](#upload-files)
2. [Submit your labeling job](#4-submit-your-labeling-job)
3. [Download the completed annotations](#5-download-the-completed-annotations)


## 1. Prepare your data for annotation
(If you haven't yet created a dataset to annotate, [follow these steps to extract frames from videos](../readme.md#extract-video-frames-for-annotation).)

In this step, we'll upload two sets of files to Amazon:
* the images you would like workers to annotate
* the html template for an annotation interface that will tell workers which body parts to label

### Create S3 buckets for your data and annotation interface

We're going to make two data storage objects, called buckets: one will host the **data to be annotated**, and the other will host an **annotation interface** with instructions for your annotators.

For the data:
1. On [AWS](http://console.aws.amazon.com), search "S3" in the Find Services menu and go to Amazon S3 (Scalable Storage in the Cloud).
2. Click the <kbd>+ Create bucket</kbd> button to make a new S3 bucket for your images. We'll be calling this `data-bucket`. ([screenshot](s3bucket.png))
3. Set a name for your bucket, and set the region to be **the same region in which you imported your Lambda functions** when you [first set up your account](readme_initialGTSetup#1-import-the-pre--and-post-processing-lambda-functions). You may leave all other default settings as is, then scroll down to click <kbd>Create bucket</kbd>.
    > Note: by default, no one other than you will be able to access images in your bucket (aside from annotators who are served the images via SageMaker.)
4. Update CORS to fix image orientation: Back in the S3 bucket list, select the S3 bucket you just created. Click <kbd>Permissions</kbd> and scroll down to the <kbd>Cross-origin resource sharing (CORS)</kbd> section. Click Edit and enter the following code in the gray box:
`[{
   "AllowedHeaders": [],
   "AllowedMethods": ["GET"],
   "AllowedOrigins": ["*"],
   "ExposeHeaders": []
}]`

For the data output:
1. Create a new S3 bucket for the annotation interface and give it a name. We'll be calling this `data-bucket-output`. Scroll down and click <kbd>Create bucket</kbd>. Once the S3 bucket is created and you are back in the Amazon S3 Bucket list page, select your S3 Bucket and update the CORS settings.

For the annotation interface:
1. Create a new S3 bucket for the annotation interface, give it a name, and *uncheck* the "Block *all* public access" box so objects in the bucket can be accessed. We'll be calling this `template-bucket`. Scroll down and click <kbd>Create bucket</kbd>. Once the S3 bucket is created and you are back in the Amazon S3 Bucket list page, select your S3 Bucket and update the CORS settings.

### Prepare your annotation interface
Now we're going to create the **annotation interface**, a simple piece of HTML with instructions for annotators.

We will create this programmatically from a configuration file called `project-config.yml` that was created inside the `project-name` directory when you [initialized your labeling project](../pose_annotation_tools#0-initialize-a-new-labeling-project).

1. Open `project-config.yml` in your text editor. If you are using the same setup as MARS, you can leave this as is, EXCEPT change the names of `data-bucket`, `template-bucket`, and `region` to reflect where you created your S3 buckets, and update the image path under "full_instructions" to include `region` and `template-bucket`.
2. From the terminal, call `python generate_AWS_template.py '/fullpath/to/project-name/project-config.yml'`

This will create a file `project-name/annotation-data/annotation-interface.template` containing your annotation template.

### Upload files
Now we're going to upload everything to your two S3 buckets.
1. Open [AWS](http://console.aws.amazon.com) and navigate to S3. First, click on the name of your `data-bucket`, then select <kbd>Upload</kbd> in the next screen.
2. Upload the images in `project-name/annotation-data/raw-images` (the images you'd like annotated).
3. Next, open your `template-bucket` and upload the interface `.template` along with any instructional images you want to include. If you are using the MARS keypoints, you can use [the instructional image provided in this repository](../annotation_interface/front_view_interface/instruction_image_bodyparts.png).

## 2. Submit your labeling job
Finally, it's time to make a labeling job- we'll do this from within a Jupyter notebook hosted on SageMaker.

1. Navigate to the SageMaker console on [AWS](http://console.aws.amazon.com), and click <kbd>Notebook</kbd>-><kbd>Notebook instances</kbd> in the left-hand menu, then click the <kbd>Create notebook instance</kbd> button.
2. Give the notebook a name, and under "Permissions and Encryption" set the IAM role to `AmazonSageMaker-ExecutionRole-xxxxxxxxxxx`. Under "Git repositories", select `Clone a public Git repository to this notebook instance only`, and add the path to this repository (http://github.com/neuroethology/MARS_developer). Finally, scroll to the bottom and click <kbd>Create notebook instance</kbd>.
3. Your new notebook should now appear in your list of notebook instances. Click <kbd>Start</kbd> under "Actions" for this notebook, and wait a few minutes for the notebook status to update from `Pending` to `InService`, then click <kbd>Open Jupyter</kbd>.
4. Navigate to [pose_annotation_tools/Submit_Labeling_Job.ipynb](../Submit_Labeling_Job.ipynb) from within the notebook session, and follow all instructions.
5. Once you have run all cells of the notebook, your job should be submitted. In the SageMaker console, click <kbd>Ground Truth</kbd>-></kbd>Labeling Jobs</kbd>, and you should see a job with name **[BUCKET NAME]-xxxxxxxxxx** in progress.

> **Note:** once the job is running, you should stop your Notebook instance (select the instance and click `Actions`>`Stop` in the Notebook instances menu), so you are not billed for leaving it running!

## 3. Download the completed annotations
When a job has finished running, its status in the Labeling Jobs menu will change to ✔️<span style="color:green">Complete</span>. Once this happens, download your completed annotations by following these steps:

1. Navigate to the S3 console on [aws](http://console.aws.amazon.com). If your images to be annotated were stored in a bucket called "BUCKET_NAME", look for the bucket **[BUCKET NAME]-output** and open it.
2. Inside **[BUCKET_NAME]-output**, open **[BUCKET_NAME]-xxxxxxxxxx** (the name of your most recent labeling job- if you have more than one, pick the one with the highest number), then open **manifests** followed by **output**. You should find a file called **output.manifest**, which contains the raw output of your labeling job.
3. Check the box next to **output.manifest**, and select <kbd>Actions</kbd>-><kbd>Download</kbd>. Save the file to `project-name/annotation-data/output.manifest`.

And you've made it! Take a deep breath, then return to Step 4 of the [MARS Pose Annotation Tools ReadMe](../../pose_annotation_tools#4-post-process-manual-pose-annotations) to clean up these annotations and visualize some example images!
