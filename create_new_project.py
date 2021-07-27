import os, sys
import pathlib
import argparse
import shutil
import glob
import re
from shutil import copytree
import requests
import zipfile
import gdown


def download_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?id=" + id
    gdown.download(URL, destination, quiet=False)


def create_new_project(location, name, download_MARS_checkpoints=True, download_demo_data=False):
    name = name.lstrip("'").rstrip("'").lstrip('"').rstrip('"') # remove quotes
    if not os.path.isdir(location):
        location = location.lstrip("'").rstrip("'") # try removing quotes
        location = location.lstrip('"').rstrip('"')
        if not os.path.isdir(location):
            print("I couldn't find the location " + location)
            return
    if os.path.isdir(os.path.join(location,name)):
        print("A project named " + name + " already exists at this location. Please delete it or choose a different name.")
        return

    # copy the config files
    project = os.path.join(location,name)
    copytree('_template', project)

    # download the model checkpoints and demo data
    if download_demo_data:
        dataset_name = 'CRIM13_sample_data'  # 2000 frames from CRIM13, manually annotated for pose
        dataset_id = '1DGUmuWgiQXM7Kx6x-QHJQathVIjQOmMR'

        print('Downloading the 2000-frame sample pose dataset (2000 manually annotated images, 289Mb)...')
        download_from_google_drive(dataset_id, os.path.join(project, dataset_name+'.zip'))
        print('  unzipping...')
        with zipfile.ZipFile(os.path.join(project, dataset_name+'.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(project))
        os.rename(os.path.join(project, dataset_name), os.path.join(project, 'annotation_data'))
        os.remove(os.path.join(project, dataset_name+'.zip')) # delete original zip file
        print('  sample dataset has been unpacked.')

    if download_MARS_checkpoints:
        ckpts_name = 'MARS_v1_8_models'
        ckpts_id = '1NyAuwI6iQdMgRB2w4zX44yFAgEkux4op'
        print('Downloading the pre-trained MARS models (2.24Gb)...')
        download_from_google_drive(ckpts_id, os.path.join(project, ckpts_name+'.zip'))
        print('  unzipping...')
        with zipfile.ZipFile(os.path.join(project, ckpts_name+'.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(project))
        
        # move checkpoints to where they need to be within the project:
        detector_black = glob.glob(os.path.join(project,ckpts_name,'detect*black*'))
        detector_white = glob.glob(os.path.join(project,ckpts_name,'detect*white*'))
        detector_resnet = glob.glob(os.path.join(project,ckpts_name,'detect*resnet*'))
        pose = glob.glob(os.path.join(project,ckpts_name,'pose*'))
        
        shutil.move(detector_black[0],os.path.join(project,'detection','black_top_model'))
        shutil.move(detector_white[0],os.path.join(project,'detection','white_top_model'))
        shutil.move(detector_resnet[0],os.path.join(project,'detection','resnet_model'))
        shutil.move(pose[0],os.path.join(project,'pose','top_model'))
        
        shutil.rmtree(os.path.join(project, ckpts_name))
        os.remove(os.path.join(project, ckpts_name+'.zip')) # delete original zip file
        print('  models have been unpacked.')

        #TODO: put the downloaded checkpoints in the right place!

    subfolders = [os.path.join(project, 'annotation_data'), os.path.join(project, 'annotation_data', 'raw_images'),
                  os.path.join(project, 'annotation_data', 'behavior_movies'), os.path.join(project, 'behavior'),
                  os.path.join(project, 'detection'), os.path.join(project, 'pose')]
    for f in subfolders:
        if not os.path.exists(f):
            os.mkdir(f)

    print("Project " + name + " created successfully.")


if __name__ == '__main__':
    """
    create_new_project command line entry point
    Arguments:
        location 		where you would like to create the project folder
        name           a name for this annotation project
    """

    parser = argparse.ArgumentParser(description='create a new annotation project', prog='new_annotation_project')
    parser.add_argument('location', type=str, help="where you would like to create the project folder")
    parser.add_argument('name', type=str, help="the name of this project")
    args = parser.parse_args(sys.argv[1:])

    create_new_project(args.location, args.name)
