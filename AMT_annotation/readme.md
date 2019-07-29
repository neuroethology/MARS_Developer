## A_annotation
This folder contains all you need to collect, extract, process frames and export information from annotation

#### AMT ANNOTATIONS

1. Make folder `keys`
2. Make folder `servers`
3. Put `mouse_pem` in `keys` folder
4. Put `aws_mous_ssh.sh` in *servers
5. `python chomod 440 mouse_pem`
5. ` make dir mount` in `servers`
6. `bash aws_mouse_ssfs.sh` editing editing the absolute path
7. `bash aws_mouse_ssh.sh`
8. Copy the images to annotation in `mount/file_to_annotate/temp`
9. Edit `config/config.py` to path where are the images to annotate
10. Copy points to click as string to instructions into `instructions.txt`
11. Create a file wehere specify how to connect points (for visualization purpose)
12. Ssh into tool folder and run `bash aws_mouse_ssh.sh`
13. Follow the remaining instruction at below.

#### PREPARE DATA FOR AMT AND EXPORT ANNOTATIONS

1. `extract_10k_frames.py` extracts frames from videos you want to annotate
2. `AWS_prep_filenames.py` saves a text file with the list of extracted frames you want to annotate
3. Follow instruction to annotate frames in AMT in B_server folder  as exlained in the previous section
4. `csv_export_correct_ann.py` exports AMT annotation to csv file correcting them if the sides or ears are flipped
5. `csv_extract_data_actions.py` extracts info from csv created exporting the annotations  and save them as a structure.
   Use `csv_extract_data_actions_allpoints.py` to extract all mouse points instead of excluding the tail
6. If you want to check the annotation and run some statitics use `plots_annotations.py`

#### FILES IN THIS FOLDER

| Filename | Description |
|----------| ------------|
|AWS_prep_filenames| write the list of filename to annotate to a text file|
|extract_10frames.py| and similar serve to extract frames you want to annotate in AMT from videos and save them as JPEG|
|export_csv_correct_ann| download annotation from AMT, correct them if there's any R/L swap and save to csv|
|extract_data_actions_csv_top.py| compute some features and save the dataset of the annotation and info of images and video to a dictionary structure ready to be converted into tf record format|
|plot_annotations.py| allows to plot the annotations and show several stats of the annotations|

#### INSTRUCTIONS

This folder contains useful tools and utilties for configuring and running
annotations. 

Here's the standard workflow for annotating a batch of JPEG frames. 

#### Assumptions

If someone has done annotations before on this server, you can probably assume
the following has been performed.

- You've set up a web-facing server and cloned this repository onto it. 
- You ran the setup scripts in `/setup/`.
- You have a domain name pointed to this server forwarding HTTPS connections,
    and have a SSL certificate configured.
- You've set up a MYSQL database and configured the connection string properly
    in `annotation_server/annotation_frontend/database.py`.
- You have an Amazon Mechanical Turk (AMT) requester account. You've configured
    an IAM user in AWS with permissions to work with the AMT account (make an
    group with permission `AmazonMechanicalTurkFullAccess`, then make a user in
    this group). 
- You've copied `annotation_server/config/crowdlib_settings.py.example` to
    `crowdlib_settings.py` and adjusted the AWS info (create a new access key in
    the AWS IAM user make above, and copy the details into
    `crowdlib_settings.py`.

You'll still need to make sure:

- You're SSHed into it and navigated to this folder (`annotation_server/tools/`). 

#### Workflow

- Prepare the data 
    - Get all the frames you want annotated as a folder of JPEG
      files. Put the files in the location specified in `config/config.py`,
      either directly in that folder or in a subfolder. 

- Configure the annotation format. Different annotation runs might want
    different keypoints annotated. The annotation version specifies a set of
    keypoints and a connection diagram to build a skeleton from the keypoints. 
    - Prepare a file INSTRUCTIONS_FILE with instructions for each point to
      click. These will be
      displayed in the GUI as a prompt for the user before each click. 
    - Prepare a file CONNECTIONS_FILE specifying the connections between clicks.
     This is used to
        generate a skeleton wireframe between the points as the user annotates.
        A proper skeleton draws lines between certain points but not others (ie,
        the end of the tail point connects to the mid-tail point but nowhere
        else). For each connection between two clicks, add a line in this file
        specifying the index of each click (0-indexed) separated by a comma. Eg.
        0,1 to connect the first and second click with a line. See
        `samples/connections.txt` for an example.
    - Select a FORMAT_ID which is a unique numeric id for this format.
    - Create a new annotation format from these files with `python
        config_annotation_format.py write FORMAT_ID -i INSTRUCTIONS_FILE -c
        CONNECTIONS_FILE -d "a description of this annotation format"`
    - Read an existing configuration with `python config_annotation_format.py
        read FORMAT_ID`.

- Configure the annotation server for your run using `config_run_parameters.py`.
    - Decide RUN_ID. This is a unique identifier for this run.
    - Decide NUM_ANNOTATIONS: how many workers should annotate each frame
    - Decide ANNOTATION_VERSION: the FORMAT_ID of the annotation format for this
        run. See above.
    - Set up the configuration using the command `python
        config_run_parameters.py write RUN_ID -na NUM_ANNOTATIONS -d "a
        description of this run" -v ANNOTATION_VERSION`.
    - Read an existing configuration with `python config_run_parameters.py
        read RUN_ID`.


- Configure which frames should be annotated for a specific RUN_ID.
    - Make a list of the filenames of every JPEG file you want annotated 
        (and which you put in the proper location above). 
        If you put the files in a subfolder, include
      the subfolder in the filename, eg. `run_5/frame.jpeg`. Save this list as a
      text file FRAMES with one filename per line. See `samples/filenames.txt` 
      for an example.
    - Add these frames to a run with the command `python config_frames.py write
        RUN_ID -f FRAMES`.
    - Read an existing configuration with `python config_frames.py
        read RUN_ID`.

- Change the active RUN_ID to the new one set up above. If the web server is
 active, this will change which
    configuration is being served immediately.
    - Run `python config_active_run_id.py write RUN_ID` to set.
    - Run `python config_active_run_id.py read` to see which RUN_ID is currently
        active.

- If not already running, start the web server by running
    `annotation_server/gunicorn.sh`. To make it run even after you disconnect
    from the SSH connection, run this command inside a `screen` instance (begin
    one by running `screen` anywhere, and leave the screen instance by typing
    CTRL-A, CTRL-D sequentially). Reconnect to an instance with `screen -r`.

- Make sure the frames are being served properly by opening the URL of your web
    server in a browser. Note any annotations you make here will be saved,
    although they will be tagged with a special tag indicating they didn't
    originate from AMT.

- Check that your Amazon Mechanical Turk is configured properly.
    - Do you have enough money in your account?

- Request annotations from AMT. 
    - Decide a price in $ per annotation PRICE. Total cost will be PRICE *
        num_annotations_per_frame * num_frames.
    - Decide a total number of annotations NUM. It's recommended to start small
        to check that the system is working properly before running everything.
        Ideally, you wouldn't request more annotations than you have images left
        to annotate, because this will leave invalid requests live on AMT,
        possibly annoying our annotators. 
    - Decide on TARGET "sandbox" or "production". Sandbox is a
        [sandbox](https://requester.mturk.com/developer/sandbox) to test things
        before requesting real people to annotate your frames. 
    - To request frames, run `python config_amt.py request TARGET -p
        PRICE -n NUM`
    - To cancel all oustanding requests for a TARGET, run `python config_amt.py
    abort TARGET`
    - To monitor the number of outstanding production requests (# requested - # complete),
        or view them as seen by annotators, go
        [here](https://www.mturk.com/mturk/searchbar?selectedSearchType=hitgroups&searchWords=%22Mouse+Image+Annotation%22&minReward=0.00&x=0&y=0). Note
        that there's some latency before changes appear here.
	- To see the equivalent information in the sandbox environment, go [here](https://workersandbox.mturk.com/mturk/searchbar?selectedSearchType=hitgroups&searchWords=%22Mouse+Image+Annotation%22&minReward=0.00&x=19&y=11).

- Monitor the number of annotations recorded in the database (displays current AMT balance, # annotations
  requested, # HITs live on AMT, # annotations recorded in database, # frames left)
	- Run `python monitor_progress.py production` to see how the current run is proceeding.
	- Run `python monitor_progress.py production -r RUN_ID` to see the status of a non-active run.

- Export the data
	- Run `python export_csv.py RUN_ID` to export the data for RUN_ID
