from __future__ import division
import os, yaml, fnmatch
import json
import copy
from pathlib import Path
import behavior_classification.annotation_parsers as map
import random
import matplotlib.pyplot as plt


def get_files(root, extensions, must_contain=''):
    all_files = []
    for ext in extensions:
        all_files.extend(Path(root).rglob('*.' + ext))
    kept_files = [i for i in all_files if must_contain in str(i.stem)]
    return kept_files


def find_videos(video_path, video_formats, must_contain=''):
    video_names = get_files(video_path + os.path.sep, video_formats, must_contain=must_contain)
    video_list = {}
    anno_list = {}
    pose_list = {}
    multi_match = {}
    no_match = []
    for v in video_names:
        # split our video filename by '_''s, search for perfect-match annotations until we run out of components:
        anno_parts = v.stem.split('_')
        anno_depth = -2
        for i in range(len(anno_parts) - 1):
            match_anno = get_files(v.parent, map.list_supported_formats(), '_'.join(anno_parts[:i + 1]))
            if len(match_anno) == 1:
                anno_depth = i
                break
            elif len(match_anno) == 0:
                match_anno = get_files(v.parent, map.list_supported_formats(), '_'.join(anno_parts[:i]))
                anno_depth = i - 1
                break
        if anno_depth == -2: # we found more than one annot file containing movie full name. Take the first one that's a perfect match.
            diffchar = []
            for a in match_anno:
                if v.stem == a.stem:
                    match_anno = [a]
                    anno_depth = len(anno_parts)
                    break
                else:
                    diffchar.append(len(a.stem.replace(v.stem, '')))
        if anno_depth == -2:  # seriously, we still don't have a match? okay, take the one that's least different.
            match_anno = [match_anno[diffchar.index(min(diffchar))]]
            anno_depth = len(anno_parts)

        pose_parts = v.stem.split('_')
        pose_depth = -2
        for i in range(len(pose_parts) - 1):
            match_pose = get_files(v.parent, ['json'], '_'.join(pose_parts[:i + 1]))
            if len(match_pose) == 1:
                pose_depth = i
                break
            elif len(match_pose) == 0:
                match_pose = get_files(v.parent, ['json'], '_'.join(pose_parts[:i]))
                pose_depth = i - 1
                break
        if pose_depth == -2:
            diffchar = []
            for a in match_pose:
                if v.stem == a.stem:
                    match_pose = [a]
                    pose_depth = len(pose_parts)
                    break
                else:
                    diffchar.append(len(a.stem.replace(v.stem, '')))
        if pose_depth == -2:  # seriously, we still don't have a match? okay, take the one that's least different.
            match_pose = [match_pose[diffchar.index(min(diffchar))]]
            pose_depth = len(pose_parts)

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


def summarize_annotations(anno_frames):
    counts = {}
    for beh in list(set(anno_frames)):
        counts[beh] = anno_frames.count(beh)
    return counts


def summarize_annotation_split(project, do_bar=False):
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
            anno_dict = map.parse_annotations(assignments[key][k][0]['anno'], omit_channels=['intruder', 'stim'])
            for entry in assignments[key][k]:
                anno_frames = [anno_dict['behs_frame'][b] for b in entry['keep_frames']]
                counts = summarize_annotations(anno_frames)
                for beh in counts.keys():
                    if beh not in behavior_time[key].keys():
                        behavior_time[key][beh] = counts[beh]
                    else:
                        behavior_time[key][beh] += counts[beh]
        master_keys += list(behavior_time[key].keys())
    master_keys = list(set(master_keys))
    master_keys.sort()

    if do_bar:
        fig, ax = plt.subplots(3, 1, figsize=[15, 15])
    else:
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

        if do_bar:
            ax[idx].bar(labels[1:], sizes[1:])
            ax[idx].tick_params(axis='x', labelrotation = 90)
        else:
            ax[idx].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
            ax[idx].axis('equal')
        ax[idx].title.set_text(key)
    fig.suptitle('Behaviors observed in train/test/validation sets')
    plt.show()
    print('list of all observed annotations:')
    print(master_keys)


def get_unique_behaviors(video_list):
    bhv_list = []
    for k in video_list.keys():
        if isinstance(video_list[k], list):
            for entry in video_list[k]:
                anno = entry['anno']
                anno_dict = map.parse_annotations(anno, omit_channels=['intruder', 'stim'])
                for ch in anno_dict['keys']:
                    bhv_list += list(anno_dict['behs_bout'][ch].keys())
        else:
            anno = video_list[k]['anno']
            anno_dict = map.parse_annotations(anno, omit_channels=['intruder', 'stim'])
            for ch in anno_dict['keys']:
                bhv_list += list(anno_dict['behs_bout'][ch].keys())

    bhv_unique = list(set(bhv_list))
    bhv_list = [i for i in bhv_unique if i != 'other']

    return bhv_list


def check_behavior_data(project):
    # matches up videos with annotations, alerts to any videos that are missing annotations or have multiple matches,
    # and prints a list of all annotated behaviors.
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    video_path = os.path.join(project, 'behavior', 'behavior_data')

    video_list, multi_match, no_match = find_videos(video_path, cfg['video_formats'])
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


def addtoset(mylist, key, entry):
    if key in mylist.keys():
        mylist[key].append(entry)
    else:
        mylist[key] = [entry]


def prep_behavior_data(project, val=0.1, test=0.2, reshuffle=True, do_bar=False, cut_videos=False, drop_label='', train_dir=[], val_dir=[], test_dir=[]):
    # shuffle data into training, validation, and test sets. Train/val/test split is currently only allowed by video.
    # passing train_dir, val_dir, and test_dir values will override the values of reshuffle/val/test
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    video_path = os.path.join(project, 'behavior', 'behavior_data')
    FR = cfg['framerate']
    video_list, _, _ = find_videos(video_path, cfg['video_formats'])
    train_list = []
    val_list = []
    test_list = []
    for d in train_dir:
        temp, _, _ = find_videos(video_path, cfg['video_formats'], must_contain=d)
        train_list += list(temp.keys())
    for d in val_dir:
        temp, _, _ = find_videos(video_path, cfg['video_formats'], must_contain=d)
        val_list += list(temp.keys())
    for d in test_dir:
        temp, _, _ = find_videos(video_path, cfg['video_formats'], must_contain=d)
        test_list += list(temp.keys())

    tMax = 0
    for video in video_list.keys():
        anno = video_list[video]['anno']
        video_list[video]['anno_dict'] = map.parse_annotations(anno, omit_channels=['intruder', 'stim'])
        if drop_label != '':  # we're using a subset of frames, figure out which ones!
            keep_frames = [i for i, j in zip(range(len(video_list[video]['anno_dict']['behs_frame'])),
                                             video_list[video]['anno_dict']['behs_frame']) if j != 'omit']
            tMax += len(keep_frames)
        else:
            tMax += video_list[video]['anno_dict']['nFrames']

    tVal = tMax * val  # minimum number of frames to assign to the validation set
    tTest = tMax * test  # minimum number of frame sto assign to the test set

    if os.path.exists(os.path.join(project, 'behavior', 'behavior_jsons', 'train_data.json')) and reshuffle:
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
            if drop_label == '':
                keep_frames = list(range(len(video_list[video]['anno_dict']['behs_frame'])))
            else:
                keep_frames = [i for i, j in zip(range(len(video_list[video]['anno_dict']['behs_frame'])),
                                                 video_list[video]['anno_dict']['behs_frame']) if j != drop_label]
            entry = {'video': video,
                     'anno': video_list[video]['anno'],
                     'pose': video_list[video]['pose'],
                     'keep_frames': keep_frames}
            if cut_videos:  # rather than splitting by video, split by 1-minute time chunks.
                chunksize = min(FR*60, tVal, tTest)
                startframes = list(range(0, len(keep_frames), chunksize))
                stopframes = list(range(chunksize, len(keep_frames), chunksize))
                if len(stopframes) < len(startframes):
                    startframes = startframes[:-1]
                    stopframes[-1] = len(keep_frames)
            else:
                startframes = [0]
                stopframes = [len(keep_frames)]
            temp = list(zip(startframes, stopframes))
            random.shuffle(temp)
            startframes, stopframes = zip(*temp)

            for count, [start, stop] in enumerate(zip(startframes, stopframes)):
                sub_entry = copy.deepcopy(entry)
                sub_entry['keep_frames'] = keep_frames[start:stop]

                if T < tVal or video in val_list:
                    addtoset(assignments['val'], str(Path(video).stem), sub_entry)
                elif T < (tVal + tTest) or video in test_list:
                    addtoset(assignments['test'], str(Path(video).stem), sub_entry)
                elif train_list == [] or video in train_list:
                    addtoset(assignments['train'], str(Path(video).stem), sub_entry)
                T += len(keep_frames[start:stop])

        if not os.path.exists(os.path.join(project, 'behavior', 'behavior_jsons')):
            os.mkdir(os.path.join(project, 'behavior', 'behavior_jsons'))
        with open(os.path.join(project, 'behavior', 'behavior_jsons', 'train_test_split.json'), 'w') as f:
            json.dump(assignments, f)

    summarize_annotation_split(project, do_bar=do_bar)


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
            anno_dict = map.parse_annotations(assignments[key][k][0]['anno'], omit_channels=['intruder', 'stim'])
            annotations = [beh_dict[b] for b in anno_dict['behs_frame']]
            with open(assignments[key][k][0]['pose']) as f:
                posedata = json.load(f)
            for entry in assignments[key][k]:
                indices = entry['keep_frames']
                saveentry = {'keypoints': [posedata['keypoints'][i] for i in indices],
                         'bbox': [posedata['bbox'][i] for i in indices],
                         'scores': [posedata['scores'][i] for i in indices],
                         'annotations': [annotations[i] for i in indices],
                         'metadata': entry}
                addtoset(savedata['sequences'][cfg['project_name']], k, saveentry)

        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_data.json'), 'w') as f:
            json.dump(savedata, f)


def set_equivalences(project, equivalences):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(project, 'behavior', 'behavior_equivalences.yaml'), 'w') as outfile:
        yaml.dump(equivalences, outfile, default_flow_style=False)
