import os, sys
import pathlib
import argparse
from shutil import copytree
import requests
import zipfile


def download_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


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

        print('  Downloading the 2000-frame sample pose dataset (2000 manually annotated images, 275Mb)...')
        download_from_google_drive(dataset_id, os.path.join(project, dataset_name+'.zip'))
        print('  unzipping...')
        with zipfile.ZipFile(os.path.join(project, dataset_name+'.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(project))
        os.rename(os.path.join(project, dataset_name), os.path.join(project, 'annotation_data'))
        print('  sample dataset has been downloaded.')

    if download_MARS_checkpoints:
        ckpts_name = 'MARS_v1_8_models'
        ckpts_id = '1N72WWzKEX0mPHzxdFuN-SffPcIBU3G5K'
        print('  Downloading the pre-trained MARS models (1.85Gb)...')
        download_from_google_drive(ckpts_id, os.path.join(project, ckpts_name+'.zip'))
        print('  unzipping...')
        with zipfile.ZipFile(os.path.join(project, ckpts_name+'.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(project))
        os.rename(os.path.join(project, ckpts_name), os.path.join(project, 'downloaded_models'))
        print('  models have been downloaded.')

        #TODO: put the downloaded checkpoints in the right place!

    subfolders = [os.path.join(project, 'annotation_data'), os.path.join(project, 'annotation_data', 'raw_images'),
                  os.path.join(project, 'annotation_data', 'behavior_movies'), os.path.join(project, 'behavior'),
                  os.path.join(project, 'detection'), os.path.join(project, 'pose')]
    for f in subfolders:
        if not os.path.exists(f):
            os.makedir(f)

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
