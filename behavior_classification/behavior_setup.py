from __future__ import division
import os, yaml, fnmatch
from pathlib import Path
import behavior_classification.annotation_parsers as map
import random
import pdb


def get_files(root, extensions, fname='*'):
    all_files = []
    for ext in extensions:
        all_files.extend(Path(root).rglob(fname + '.' + ext))
    return all_files


def find_videos(project):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    video_path = os.path.join(project, 'behavior', 'behavior_data')
    video_names = get_files(video_path + os.path.sep, cfg['video_formats'])
    video_list = {}
    anno_list = {}
    pose_list = {}
    multi_match = {}
    no_match = []
    for v in video_names:
        # split our video filename by '_''s, search for perfect-match annotations until we run out of components:
        anno_parts = v.stem.split('_')
        for i in range(len(anno_parts) - 1):
            match_anno = get_files(v.parent, map.list_supported_formats(), '_'.join(anno_parts[:i + 1]) + '*')
            if len(match_anno) == 1:
                anno_depth = i
                break
            elif len(match_anno) == 0:
                match_anno = get_files(v.parent, map.list_supported_formats(), '_'.join(anno_parts[:i]) + '*')
                anno_depth = i - 1
                break

        pose_parts = v.stem.split('_')
        for i in range(len(pose_parts) - 1):
            match_pose = get_files(v.parent, ['json'], '_'.join(pose_parts[:i + 1]) + '*')
            if len(match_pose) == 1:
                pose_depth = i
                break
            elif len(match_pose) == 0:
                match_pose = get_files(v.parent, ['json'], '_'.join(pose_parts[:i]) + '*')
                pose_depth = i - 1
                break

        if match_anno and match_pose: # greedy matching of annotation and pose files to a behavior movie:
            if len(match_anno) == 1 and len(match_pose) == 1:  # single match, keep it for now
                video_list[str(v)] = {'anno': str(match_anno[0]), 'anno_depth': anno_depth,
                                      'pose': str(match_pose[0]), 'pose_depth': pose_depth}
                if str(match_anno[0]) in anno_list.keys():
                    anno_list[str(match_anno[0])].append(str(v))
                else:
                    anno_list[str(match_anno[0])] = [str(v)]
                if str(match_pose[0]) in pose_list.keys():
                    pose_list[str(match_pose[0])].append(str(v))
                else:
                    pose_list[str(match_pose[0])] = [str(v)]
            else:
                multi_match[str(v)] = {'anno': match_anno, 'anno_depth': anno_depth,
                                       'pose': match_pose, 'pose_depth': pose_depth}
        else:
            no_match.append(str(v))

        for anno in anno_list.keys(): #check to see if multiple videos ever got mapped to the same file
            if len(anno_list[anno]) > 1:
                anno_depths = []
                for v in anno_list[anno]:
                    if v in video_list.keys():  # make sure it hasn't already been deleted
                        anno_depths.append(video_list[v]['anno_depth'])
                if len(anno_depths) == 0:
                    continue
                max_depth = max(anno_depths)

                # if two videos matched an annotation equally well, neither one gets it
                if anno_depths.count(max_depth) > 1 or len(anno_depths) <= 1:
                    max_depth += 1
                for v in anno_list[anno]:  # remove poor matches
                    if v in video_list.keys():  # make sure it hasn't already been deleted
                        if video_list[v]['anno_depth'] != max_depth:
                            del video_list[v]
                            for a in anno_list.keys():
                                if v in anno_list[a]:
                                    anno_list[a].remove(v)
                            for p in pose_list.keys():
                                if v in pose_list[p]:
                                    pose_list[p].remove(v)
                            no_match.append(str(v))

        for pose in pose_list.keys():  # this is the exact same as the above, just for pose files (yes, this should be a function instead.)
            if len(pose_list[pose]) > 1:
                pose_depths = []
                for v in pose_list[pose]:
                    pose_depths.append(video_list[v]['pose_depth'])
                if len(pose_depths) == 0:
                    continue
                max_depth = max(pose_depths)

                if pose_depths.count(max_depth) > 1 or len(pose_depths) <= 1:
                    max_depth += 1
                for v in pose_list[pose]:  # remove poor matches
                    if v in video_list.keys():  # make sure it hasn't already been deleted
                        if video_list[v]['pose_depth'] != max_depth:
                            del video_list[v]
                            for a in anno_list.keys():
                                if v in anno_list[a]:
                                    anno_list[a].remove(v)
                            for p in pose_list.keys():
                                if v in pose_list[p]:
                                    pose_list[p].remove(v)
                            no_match.append(str(v))




    return video_list, multi_match, no_match


def check_behavior_data(project):
    # matches up videos with annotations, alerts to any videos that are missing annotations or have multiple matches,
    # and prints a list of all annotated behaviors.
    video_list, multi_match, no_match = find_videos(project)
    total_vids = len(no_match) + len(video_list.keys()) + len(multi_match.keys())

    print('Processing behavior annotations in project ' + project)
    print('  ' + str(total_vids) + ' total videos found.')
    if len(video_list.keys()) == total_vids:
        print('  all videos successfully matched with annotation and pose files!')

    bhv_list = []
    for video in video_list.keys():
        anno = video_list[video]['anno']
        anno_dict = map.parse_annotations(anno)
        for ch in anno_dict['keys']:
            bhv_list += list(anno_dict['behs_bout'][ch].keys())

    bhv_unique = list(set(bhv_list))
    bhv_list = [i for i in bhv_unique if i != 'other']
    print('List of behavior annotations found:')
    for b in bhv_list:
        print('  ' + b)

    if no_match:
        print('--')
        print('Warning: annotations and/or pose estimates could not be found for the following videos (' +
              str(len(no_match)) + '/' + str(total_vids) + '):')
        for v in no_match:
            print('  ' + v)

    if multi_match:
        print('--')
        print('Warning: multiple annotations and/or pose estimates were matched to the following videos (' +
              str(len(multi_match.keys())) + '/' + str(total_vids) + '):')
        for v in multi_match.keys():
            print('  ' + v)
            for anno in multi_match[v]['anno']:
                print('    annot:  ' + str(anno.name))
            for pose in multi_match[v]['pose']:
                print('    pose:   ' + str(pose.name))

    if no_match or multi_match:
        print('--')
        print('Only videos with successfully matched annotations/pose files will be used for classification (' +
              str(len(video_list.keys())) + '/' + str(total_vids) + ')')


def prep_behavior_data(project, val=0.1, test=0.2, reshuffle=True):
    # shuffle data into training, validation, and test sets. Train/val/test split is currently only allowed by video.
    video_list, _, _ = find_videos(project)

    tMax = 0
    for video in video_list.keys():
        anno = video_list[video]['anno']
        video_list[video]['anno_dict'] = map.parse_annotations(anno)
        tMax += video_list[video]['anno_dict']['nFrames']

    tVal = tMax * val    # minimum number of frames to assign to the validation set
    tTest = tMax * test  # minimum number of frame sto assign to the test set

    if reshuffle or not os.path.exists(os.path.join(project,'behavior','train_test_split.json')):
        if not reshuffle:
            print('Couldn''t find a saved train/test split, overriding reshuffle=False argument.')
        keys = list(video_list)
        random.shuffle(keys)
        T = 0
        for video in keys:
            if T < tVal:
                # assign to val
            elif T < (tVal + tTest):
                # assign to test
            else:
                # assign to train


def set_equivalences(project, equivalences):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(project, 'behavior', 'behavior_equivalences.yaml'), 'w') as outfile:
        yaml.dump(equivalences, outfile, default_flow_style=False)
