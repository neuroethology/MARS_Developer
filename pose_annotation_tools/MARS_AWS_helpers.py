import os
import boto3
import sagemaker


def create_manifest(bucket_name,manifest_name='file_list.manifest'):
    filelist = []
    s3r = boto3.resource('s3')
    s3 = boto3.client('s3')
    
    # Use s3 resource to pull and append all the file names in the bucket
    bucket = s3r.Bucket(bucket_name)
    files_in_bucket = list(bucket.objects.all())
    
    for i in range(len(files_in_bucket)):
        file_name = files_in_bucket[i].key
        if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith('tiff'):
            filelist.append('{"source-ref":"s3://' + bucket_name + '/' + file_name + '"}')
            
    # next step, convert filelist to binary data and call put_object
    if not manifest_name.endswith('.manifest'):
        manifest_name = manifest_name + '.manifest'

    with open('tmp_'+manifest_name,'w') as f:
        for item in filelist:
            f.write("%s\n" % item)
    
    # Use s3 client to create the manifest file in s3
    s3.put_object(Body=open('tmp_'+manifest_name, 'rb'), Bucket=bucket_name, Key=manifest_name)
    print('created a manifest called "' + manifest_name + '" in ' + bucket_name)
    
    return manifest_name.replace('.manifest','')


def check_bucket_region(role, task):
    region = boto3.session.Session().region_name
    s3 = boto3.client('s3')
    bucket_region = s3.head_bucket(Bucket=task['BUCKET'])['ResponseMetadata']['HTTPHeaders']['x-amz-bucket-region']
    assert bucket_region == region, "You S3 bucket {} and this notebook need to be in the same region.".format(task['BUCKET'])


def configure_workforce(task):
    # Info about the workforce and task duration.
    human_task_config = {
          "AnnotationConsolidationConfig": { "AnnotationConsolidationLambdaArn": task['arns']['acs_arn'], },
          "PreHumanTaskLambdaArn": task['arns']['prehuman_arn'],
          "MaxConcurrentTaskCount": 200,  # number of images that will be sent at a time to the workteam.
          "NumberOfHumanWorkersPerDataObject": task['price']['num_workers'],
          "TaskAvailabilityLifetimeInSeconds": 12*60*60,  # seconds to complete all pending tasks.
          "TaskDescription": task['info']['task_description'],
          "TaskKeywords": task['info']['task_keywords'],
          "TaskTimeLimitInSeconds": 300,  # Each image must be labeled within 5 minutes.
          "TaskTitle": task['info']['task_title'],
          "UiConfig": {"UiTemplateS3Uri": task['UITEMPLATE'], }
        }

    # Specifies the workforce and compensation.
    human_task_config["PublicWorkforceTaskPrice"] = {
                "AmountInUsd": {
                   "Dollars": task['price']['dollars'],
                   "Cents": task['price']['cents'],
                   "TenthFractionsOfACent": task['price']['tenthcent'],
                }
            }
    human_task_config["WorkteamArn"] = task['arns']['workteam_arn']

    return human_task_config


def configure_ground_truth(task, human_task_config, role):
    # Info about the task dataset.
    ground_truth_request = {
            "InputConfig" : {
              "DataSource": {
                "S3DataSource": {
                  "ManifestS3Uri": 's3://{}/{}.manifest'.format(task['BUCKET'],task['MANIFEST']),
                }
              },
              "DataAttributes": {
                "ContentClassifiers": [
                  "FreeOfPersonallyIdentifiableInformation",
                  "FreeOfAdultContent"
                ]
              },
            },
            "OutputConfig" : {
              "S3OutputPath": 's3://{}-output/'.format(task['BUCKET']),
            },
            "HumanTaskConfig" : human_task_config,
            "LabelingJobName": task['info']['job_name'],
            "RoleArn": role,
            "LabelAttributeName": "annotatedResult",
        }

    return ground_truth_request
