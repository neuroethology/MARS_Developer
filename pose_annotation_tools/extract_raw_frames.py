import os, sys
import pathlib
import argparse
from shutil import copyfile
import extract_frames as ef


def extract_raw_frames(input_dir, project_path, n_frames, to_skip):
    if not os.path.isdir(project_path):
        project_path = project_path.lstrip("'").rstrip("'").lstrip('"').rstrip('"')
        if not os.path.isdir(project_path):
            print("Couldn't find a project at " + project_path)
            return
    ef.extract_frames(input_dir, os.path.join(project_path,'annotation_data','raw_images'), n_frames, to_skip)
    print("Added " + str(n_frames) " frames to " os.path.basename(project_path))


if __name__ ==  '__main__':
    """
    extract_raw_frames command line entry point
    Arguments:
        input_dir           Directory/folder to look for video files to extract frames from
        project             Full path to your labeling project
        frames_to_extract   Total number of frames to extract across all videos found
        frames_to_skip      Number of frames to skip at the beginning of each video. Default is 0.
    """

    parser = argparse.ArgumentParser(description='extract_frames command line', prog='extract_frames')
    parser.add_argument('input_dir', type=str, help="directory/folder to look for video files to extract frames from")
    parser.add_argument('project', type=str, help="Full path to your labeling project")
    parser.add_argument('frames_to_extract', type=int, help="total number of frames to extract across all videos found")
    parser.add_argument('-s', '--skip', dest='frames_to_skip', type=int, default=0, help="number of frames to skip at the beginning of each video")
    args = parser.parse_args(sys.argv[1:])

    extract_frames(args.input_dir, args.project, args.frames_to_extract, args.frames_to_skip)
