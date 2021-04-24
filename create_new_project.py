import os, sys
import argparse
from shutil import copytree
import requests


def download_CRIM13_demo_from_google_drive(id, destination):
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
            if chunk: # filter out keep-alive new chunks
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
        dataset_id = '1J73k-RC1CyJQOjUdWr-75P3w_mfpRvXr'

        print('  Downloading the 2000-frame sample pose dataset (2000 manually annotated images, 283Mb)...')
        download_CRIM13_demo_from_google_drive(dataset_id, project)
        print("  unpacking...")
        os.rename(os.path.join(project, dataset_name), os.path.join(project, 'annotation_data'))
        os.mkdir(os.path.join(project, 'behavior'))
        print('  done.')

    if download_MARS_checkpoints:
        print('  Checkpoints are not online yet, try again later')

    if not os.path.exists(os.path.join(project, 'annotation_data')):  # empty folders don't clone?
        os.mkdir(os.path.join(project, 'annotation_data'))
        os.mkdir(os.path.join(project, 'annotation_data', 'raw_images'))
        os.mkdir(os.path.join(project, 'annotation_data', 'behavior_movies'))
        os.mkdir(os.path.join(project, 'behavior'))


    # os.mkdir(os.path.join(project,'detection'))
    # os.mkdir(os.path.join(project,'pose'))
    # os.mkdir(os.path.join(project,'behavior'))
    # config = os.path.join('_templates','project_config.yaml')

    print("Project " + name + " created successfully.")


if __name__ ==  '__main__':
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
