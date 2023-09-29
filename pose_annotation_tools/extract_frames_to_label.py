import sys
from os import path
import argparse
from shutil import copyfile
sys.path.append('../')
from Util import extract_frames as ef
import re
import logging


def extract_frames_to_label(project_path, n_frames=1000, to_skip=0, verbosity='0'):
    """
    extracts frames from videos in a provided path for future labeling.
    
        project_path        base directory of project (created using MARS create_new_project)
        n_frames            *total* number of frames to extract across all videos
        to_skip             Number of frames to skip at the beginning of each video
        verbosity           amount of information to share (default:warnings and errors only)
    
    """

    if verbosity.lower() == 'info':
        logging.basicConfig(format="%(funcName)s:: %(message)s", level=logging.INFO)
    elif verbosity.lower() == 'debug': 
        logging.basicConfig(format="%(funcName)s:: %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(funcName)s:: %(message)s", level=logging.WARNING)

    # does the project exist?
    if not path.isdir(project_path):
        project_path = re.sub('\'|\"',r'',project_path)
        if not path.isdir(project_path):
            logging.warning("Couldn't find a project at " + project_path)
            return
    
    # parse and pass everything to Util function extract_frames
    input_dir = path.join(project_path,'annotation_data','behavior_movies') 
    output_dir = path.join(project_path,'annotation_data','raw_images')
    ef.extract_frames(input_dir, output_dir, n_frames, to_skip, verbosity=verbosity)
    print(f"Added frames to {project_path}")


if __name__ ==  '__main__':
    """
    extract_frames_to_label command line entry point
    Arguments:
        input_dir           Directory/folder to look for video files to extract frames from
        project             Full path to your labeling project
        frames_to_extract   Total number of frames to extract across all videos found
        frames_to_skip      Number of frames to skip at the beginning of each video. Default is 0.
    """

    parser = argparse.ArgumentParser(description='extract_frames command line', prog='extract_frames')
    parser.add_argument('project', type=str, help="Full path to your labeling project")
    parser.add_argument('frames_to_extract', type=int, default=1000, help="total number of frames to extract across all videos found")
    parser.add_argument('-s', '--skip', dest='frames_to_skip', type=int, default=0, help="number of frames to skip at the beginning of each video")
    args = parser.parse_args(sys.argv[1:])

    extract_frames_to_label(args.project, args.frames_to_extract, args.frames_to_skip)
