## Create a private team for your task
>These instructions are reproduced from the [Ground Truth object detection tutorial](https://github.com/aws/amazon-sagemaker-examples/tree/master/ground_truth_labeling_jobs/ground_truth_object_detection_tutorial) in the [Amazon SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples/) GitHub Repo. All instructions should be carried out in the [AWS Console](http://console.aws.amazon.com).

We will create a `private workteam` and add only one user (you) to it. Then, we will modify the Ground Truth API job request to send the task to that workforce. You will then be able to see your annotation job exactly as the public annotators would see it. You could even annotate the whole dataset yourself!

To create a private team:
1. Go to `AWS Console > Amazon SageMaker > Labeling workforces`
2. Click <kbd>Private</kbd> and then <kbd>Create private team</kbd>.
3. Enter the desired name for your private workteam.
4. Select <kbd>Create a new Amazon Cognito user group</kbd> and click <kbd>Create private team.</kbd>
5. The AWS Console should now return to `AWS Console > Amazon SageMaker > Labeling workforces`.
6. Click on <kbd>Invite new workers</kbd> in the <kbd>Workers</kbd> tab.
7. Enter your own email address in the <kbd>Email addresses</kbd> section and click <kbd>Invite new workers.</kbd>
8. Click on your newly created team under the <kbd>Private teams</kbd> tab.
9. Select the <kbd>Workers</kbd> tab and click <kbd>Add workers to team.</kbd>
10. Select your email and click <kbd>Add workers to team.</kbd>
11. The AWS Console should again return to `AWS Console > Amazon SageMaker > Labeling workforces`. Your newly created team should be visible under <kbd>Private teams</kbd>. Next to it you will see an `ARN` which is a long string that looks like `arn:aws:sagemaker:region-name-123456:workteam/private-crowd/team-name`. Copy this ARN into the cell below.
12. You should get an email from `no-reply@verificationemail.com` that contains your workforce username and password.
13. In `AWS Console > Amazon SageMaker > Labeling workforces > Private`, click on the URL under `Labeling portal sign-in URL`. Use the email/password combination from the previous step to log in (you will be asked to create a new, non-default password).

That's it! You can invite your colleagues to participate in the labeling job by clicking the <kbd>Invite new workers</kbd> button.

The [SageMaker Ground Truth documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-workforce-management-private.html) has more details on the management of private workteams.
