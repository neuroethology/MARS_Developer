import os, sys
import pathlib
import argparse
from shutil import copyfile


def new_annotation_project(location,name):
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
    project = os.path.join(location,name)
    os.mkdir(project)
    os.mkdir(os.path.join(project,'annotation_data'))
    os.mkdir(os.path.join(project,'annotation_data','raw_images'))
    config = os.path.join(pathlib.Path(__file__).parent.absolute(),'annot_config.yml')
    copyfile(config,os.path.join(project,'annotation_data','annot_config.yml'))
    print("Project " + name + " created successfully.")


if __name__ ==  '__main__':
    """
    new_annotation_project command line entry point
    Arguments:
        location 		where you would like to create the project folder
        name           a name for this annotation project
    """

    parser = argparse.ArgumentParser(description='create a new annotation project', prog='new_annotation_project')
    parser.add_argument('location', type=str, help="where you would like to create the project folder")
    parser.add_argument('name', type=str, help="the name of this project")
    args = parser.parse_args(sys.argv[1:])

    new_annotation_project(args.location, args.name)
