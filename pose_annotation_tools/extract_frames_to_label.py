import os, sys
import pathlib
import argparse
from shutil import copyfile
sys.path.append('../')
from Util import extract_frames as ef


def extract_frames_to_label(project_path, n_frames=1000, to_skip=0):
    if not os.path.isdir(project_path):
        project_path = project_path.lstrip("'").rstrip("'").lstrip('"').rstrip('"')
        if not os.path.isdir(project_path):
            print("Couldn't find a project at " + project_path)
            return
    input_dir = os.path.join(project_path,'annotation_data','behavior_movies')
    output_dir = os.path.join(project_path,'annotation_data','raw_images')
    ef.extract_frames(input_dir, output_dir, n_frames, to_skip)
    print("Added frames to " + os.path.basename(project_path))


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
