from __future__ import division
import os, yaml, fnmatch
import json
from pathlib import Path
import behavior_classification.annotation_parsers as map
import random
import matplotlib.pyplot as plt


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

        if match_anno and match_pose:  # greedy matching of annotation and pose files to a behavior movie:
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

        for anno in anno_list.keys():  # check to see if multiple videos ever got mapped to the same file
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


def summarize_annotations(anno_dict):
    counts = {}
    for beh in list(set(anno_dict['behs_frame'])):
        counts[beh] = anno_dict['behs_frame'].count(beh)
    return counts


def summarize_annotation_split(project):
    splitfile = os.path.join(project, 'behavior', 'behavior_jsons', 'train_test_split.json')
    if not os.path.exists(splitfile):
        print('summarize_annotation_split failed, couldn\'t find train_test_split.json in the project directory.')
        return

    with open(splitfile) as f:
        assignments = json.load(f)

    behavior_time = {'train': {}, 'test': {}, 'val': {}}
    master_keys = []
    for idx, key in enumerate(['train', 'test', 'val']):
        for k in assignments[key].keys():
            anno_dict = map.parse_annotations(assignments[key][k]['anno'])
            counts = summarize_annotations(anno_dict)
            for beh in counts.keys():
                if beh not in behavior_time[key].keys():
                    behavior_time[key][beh] = counts[beh]
                else:
                    behavior_time[key][beh] += counts[beh]
        master_keys += list(behavior_time[key].keys())
    master_keys = list(set(master_keys))
    master_keys.sort()

    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    for idx, key in enumerate(['train', 'test', 'val']):
        if 'other' in behavior_time[key].keys():
            sizes = [behavior_time[key]['other']]
            labels = ['other']
            explode = [0.1]
        else:
            sizes = []
            labels = []
            explode = []
        for beh in master_keys:
            if beh != 'other':
                if beh in behavior_time[key].keys():
                    sizes.append(behavior_time[key][beh])
                else:
                    sizes.append(0)
                labels.append(beh)
                explode.append(0)

        ax[idx].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        ax[idx].axis('equal')
        ax[idx].title.set_text(key)
    fig.suptitle('Behaviors observed in train/test/validation sets')

    plt.show()


def get_unique_behaviors(video_list):
    bhv_list = []
    for k in video_list.keys():
        anno = video_list[k]['anno']
        anno_dict = map.parse_annotations(anno)
        for ch in anno_dict['keys']:
            bhv_list += list(anno_dict['behs_bout'][ch].keys())

    bhv_unique = list(set(bhv_list))
    bhv_list = [i for i in bhv_unique if i != 'other']

    return bhv_list


def check_behavior_data(project):
    # matches up videos with annotations, alerts to any videos that are missing annotations or have multiple matches,
    # and prints a list of all annotated behaviors.
    video_list, multi_match, no_match = find_videos(project)
    total_vids = len(no_match) + len(video_list.keys()) + len(multi_match.keys())

    print('Processing behavior annotations in project ' + project)
    print('  ' + str(total_vids) + ' total videos found.')
    if len(video_list.keys()) == total_vids:
        print('  all videos successfully matched with annotation and pose files!')

    bhv_list = get_unique_behaviors(video_list)
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

    tVal = tMax * val  # minimum number of frames to assign to the validation set
    tTest = tMax * test  # minimum number of frame sto assign to the test set

    if os.path.exists(os.path.join(project, 'behavior', 'behavior_jsons', 'train_data.json')):
        a = 'x'
        while not a.lower() in ['y', 'n']:
            a = input('Delete existing train/test splits and reshuffle? (y/n)')
        if a.lower() == 'y':
            os.remove(os.path.join(project, 'behavior', 'behavior_jsons', 'train_data.json'))
            os.remove(os.path.join(project, 'behavior', 'behavior_jsons', 'test_data.json'))
            os.remove(os.path.join(project, 'behavior', 'behavior_jsons', 'val_data.json'))
        else:
            reshuffle = False

    if reshuffle or not os.path.exists(os.path.join(project, 'behavior', 'behavior_jsons', 'train_test_split.json')):
        if not reshuffle:
            print('Couldn\'t find a saved train/test split, overriding reshuffle=False argument.')
        keys = list(video_list)
        random.shuffle(keys)
        T = 0
        assignments = {'train': {}, 'test': {}, 'val': {}}
        for video in keys:
            entry = {'video': video,
                     'anno': video_list[video]['anno'],
                     'pose': video_list[video]['pose']}
            if T < tVal:  # assign to val
                assignments['val'][str(Path(video).stem)] = entry
            elif T < (tVal + tTest):  # assign to test
                assignments['test'][str(Path(video).stem)] = entry
            else:  # assign to train
                assignments['train'][str(Path(video).stem)] = entry

            T += video_list[video]['anno_dict']['nFrames']

        if not os.path.exists(os.path.join(project, 'behavior', 'behavior_jsons')):
            os.mkdir(os.path.join(project, 'behavior', 'behavior_jsons'))
        with open(os.path.join(project, 'behavior', 'behavior_jsons', 'train_test_split.json'), 'w') as f:
            json.dump(assignments, f)

    summarize_annotation_split(project)


def apply_clf_splits(project):
    splitfile = os.path.join(project, 'behavior', 'behavior_jsons', 'train_test_split.json')
    if not os.path.exists(splitfile):
        print('apply_clf_splits failed, please run prep_behavior_data first.')
        return
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(splitfile) as f:
        assignments = json.load(f)

    behs = []
    for key in ['train', 'test', 'val']:
        behs += get_unique_behaviors(assignments[key])
    behs = list(set(behs))
    beh_dict = {'other': 0}
    for i, b in enumerate(behs):
        beh_dict[b] = i + 1

    for idx, key in enumerate(['train', 'test', 'val']):
        print('saving ' + key + ' set...')
        savedata = {'vocabulary': beh_dict, 'sequences': {cfg['project_name']: {}}}
        keylist = list(assignments[key].keys())
        for k in keylist:
            anno_dict = map.parse_annotations(assignments[key][k]['anno'])
            annotations = [beh_dict[b] for b in anno_dict['behs_frame']]

            with open(assignments[key][k]['pose']) as f:
                posedata = json.load(f)

            entry = {'keypoints': posedata['keypoints'],
                     'bbox': posedata['bbox'],
                     'scores': posedata['scores'],
                     'annotations': annotations,
                     'metadata': assignments[key][k]}

            savedata['sequences'][cfg['project_name']][k] = entry
        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_data.json'),'w') as f:
            json.dump(savedata, f)


def set_equivalences(project, equivalences):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(project, 'behavior', 'behavior_equivalences.yaml'), 'w') as outfile:
        yaml.dump(equivalences, outfile, default_flow_style=False)
