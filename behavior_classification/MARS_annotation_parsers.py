from __future__ import division
import numpy as np
import pdb


def parse_annotations(fid, use_channels=[], timestamps=[]):
    # use this function to load annotations of either file type!
    if fid.endswith('.txt'):
        ann_dict = parse_txt(fid)
        return ann_dict
    elif fid.endswith('.annot'):
        ann_dict = parse_annot(fid, use_channels, timestamps)
        return ann_dict


def parse_txt(f_ann):
    header='Caltech Behavior Annotator - Annotation File'
    conf = 'Configuration file:'
    fid = open(f_ann)
    ann = fid.read().splitlines()
    fid.close()
    NFrames = []
    # check the header
    assert ann[0].rstrip() == header
    assert ann[1].rstrip() == ''
    assert ann[2].rstrip() == conf
    # parse action list
    l = 3
    names = [None] *1000
    keys = [None] *1000
    k = -1

    #get config keys and names
    while True:
        ann[l] = ann[l].rstrip()
        if not isinstance(ann[l], str) or not ann[l]:
            l+=1
            break
        values = ann[l].split()
        k += 1
        names[k] = values[0]
        keys[k] = values[1]
        l += 1
    names = names[:k+1]
    keys = keys[:k+1]

    #read in each stream in turn until end of file
    bnds0 = [None]*10000
    types0 = [None]*10000
    actions0 = [None]*10000
    nStrm1 = 0
    while True:
        ann[l] = ann[l].rstrip()
        nStrm1 += 1
        t = ann[l].split(":")
        l += 1
        ann[l] = ann[l].rstrip()
        assert int(t[0][1]) == nStrm1
        assert ann[l] == '-----------------------------'
        l += 1
        bnds1 = np.ones((10000,2),dtype=int)
        types1 = np.ones(10000,dtype=int)*-1
        actions1 = [None] *10000
        k = 0
        # start the annotations
        while True:
            ann[l] = ann[l].rstrip()
            t = ann[l]
            if not isinstance(t, str) or not t:
                l += 1
                break
            t = ann[l].split()
            type = [i for i in range(len(names)) if  t[2]== names[i]]
            type = type[0]
            if type == None:
                print('undefined behavior' + t[2])
            if bnds1[k-1,1] != int(t[0])-1 and k>0:
                print('%d ~= %d' % (bnds1[k,1], int(t[0]) - 1))
            bnds1[k,:] = [int(t[0]), int(t[1])]
            types1[k] = type
            actions1[k] = names[type]
            k += 1
            l += 1
            if l == len(ann):
                break
        if nStrm1 == 1:
            nFrames = bnds1[k-1,1]
        assert nFrames == bnds1[k-1,1]
        bnds0[nStrm1-1] = bnds1[:k]
        types0[nStrm1-1] = types1[:k]
        actions0[nStrm1-1] = actions1[:k]
        if l == len(ann):
            break
        while not ann[l]:
            l += 1

    bnds = bnds0[:nStrm1]
    types = types0[:nStrm1]
    actions = actions0[:nStrm1]

    idx = 0
    if len(actions[0])< len(actions[1]):
        idx = 1
    type_frame = []
    action_frame = []
    action_bouts = {'Ch1':{}}
    len_bnd = []

    for i in range(len(bnds[idx])):
        numf = bnds[idx][i,1] - bnds[idx][i,0]+1
        len_bnd.append(numf)
        action_frame.extend([actions[idx][i]] * numf)
        if actions[idx][i] not in action_bouts['Ch1'].keys():
            action_bouts['Ch1'][actions[idx][i]] = [bnds[idx][i,:]]
        else:
            action_bouts['Ch1'][actions[idx][i]] = np.vstack((action_bouts['Ch1'][actions[idx][i]],bnds[idx][i,:]))
        type_frame.extend([types[idx][i]] * numf)

    ann_dict = {
        'keys': ['Ch1'],
        'behs': names,
        'nstrm': nStrm1,
        'nFrames': nFrames,
        'behs_se': bnds,
        'behs_dur': len_bnd,
        'behs_bout': action_bouts,
        'behs_frame': action_frame
    }

    return ann_dict


def parse_annot(filename, use_channels = [], timestamps = []):
    """ Takes as input a path to a .annot file and returns the frame-wise behavioral labels. Optional input use_channels
    only returns annotations in the specified channel(s); default behavior is to merge all channels. Passing timestamps
    from a seq movie will make sure that annotated times are converted to frame numbers correctly in the instance where
    some frames are dropped."""
    if not filename:
        print("No filename provided")
        return -1

    behaviors = []
    channel_names = []
    keys = []

    channel_dict = {}
    bouts_dict = {}
    with open(filename, 'r') as annot_file:
        line = annot_file.readline().rstrip()
        # Parse the movie files
        while line != '':
            line = annot_file.readline().rstrip()
            # Get movie files if you want

        # Parse the stim name and other stuff
        line = annot_file.readline().rstrip()
        split_line = line.split()
        stim_name = split_line[-1]

        line = annot_file.readline().rstrip()
        split_line = line.split()
        start_frame = int(split_line[-1])

        line = annot_file.readline().rstrip()
        split_line = line.split()
        end_frame = int(split_line[-1])

        framerate = 30 # provide a default framerate if the annot file doesn't have one
        line = annot_file.readline().rstrip()
        if(not(line == '')): # newer annot files have a framerate line added
            split_line = line.split()
            framerate = float(split_line[-1])
            line = annot_file.readline().rstrip()
        assert (line == '')

        # Just pass through whitespace
        while line == '':
            line = annot_file.readline().rstrip()

        # At the beginning of list of channels
        assert 'channels' in line
        line = annot_file.readline().rstrip()
        while line != '':
            key = line
            keys.append(key)
            bouts_dict[key] = {}
            line = annot_file.readline().\
                rstrip()

        # At beginning of list of annotations.
        line = annot_file.readline().rstrip()
        assert 'annotations' in line
        line = annot_file.readline().rstrip()
        while line != '':
            behavior = line
            behaviors.append(behavior)
            line = annot_file.readline().rstrip()

        # At the start of the sequence of channels
        line = annot_file.readline()
        while line != '':
            # Strip the whitespace.
            line = line.rstrip()

            assert ('----------' in line)
            channel_name = line.rstrip('-')
            channel_name = channel_name[:3] # sloppy fix for now, to get simplified channel name-----------------------
            channel_names.append(channel_name)

            behaviors_framewise = [''] * end_frame
            behaviors_boutwise = {}
            line = annot_file.readline().rstrip()
            while '---' not in line:

                # If we've reached EOF (end-of-file) break out of this loop.
                if line == '':
                    break

                # Now get rid of newlines and trailing spaces.
                line = line.rstrip()

                # If this is a blank
                if line == '':
                    line = annot_file.readline()
                    continue

                # Now we're parsing the behaviors
                if '>' in line:
                    curr_behavior = line[1:]
                    behaviors_boutwise[curr_behavior] = np.empty((0,2),int)
                    # Skip table headers.
                    annot_file.readline()
                    line = annot_file.readline().rstrip()

                # Split it into the relevant numbers
                start_stop_duration = line.split()

                # Collect the bout info.
                # parse bouts that are in frames
                if all('.' not in s for s in start_stop_duration):
                    bout_start    = max((int(start_stop_duration[0]),start_frame-1))
                    bout_end      = min((int(start_stop_duration[1]),end_frame-start_frame+1))
                    bout_duration = int(start_stop_duration[2])-1
                elif len(timestamps) != 0:
                    bout_start    = max((np.where(np.append(timestamps,np.inf) >= float(start_stop_duration[0]))[0][0],start_frame-1))
                    bout_end      = min((np.where(np.append(timestamps,np.inf) >= float(start_stop_duration[1]))[0][0],end_frame-start_frame+1))
                    bout_duration = bout_end-bout_start
                else:
                    bout_start = max((int(round(float(start_stop_duration[0])*framerate)),start_frame-1))
                    bout_end = min((int(round(float(start_stop_duration[1])*framerate)),end_frame-start_frame+1))
                    bout_duration = bout_end-bout_start

                # Store it in the appropriate place.
                behaviors_boutwise[curr_behavior] = np.vstack((behaviors_boutwise[curr_behavior],np.array([bout_start, bout_end],int)))
                if(bout_start <= end_frame):
                    behaviors_framewise[(bout_start-1):bout_end] = [curr_behavior] * (bout_duration+1)

                line = annot_file.readline()

                # end of channel
            behaviors_boutwise['other'] = np.array([[1, end_frame]],int)
            channel_dict[channel_name] = behaviors_framewise
            bouts_dict[channel_name] = behaviors_boutwise
        
        changed_behavior_list = merge_channels(bouts_dict, use_channels, end_frame)

        ann_dict = {
            'keys': keys,
            'behs': behaviors,
            'nstrm': len(channel_names),
            'nFrames': end_frame,
            'behs_frame': changed_behavior_list,
            'behs_bout': bouts_dict
        }
        return ann_dict


def rast_to_bouts(oneHot, names): # a helper for the bento save format
    bouts = dict.fromkeys(names, None)
    for val,name in enumerate(names):
        bouts[name] = {'start': [], 'stop': []}
        rast = [annot == val+1 for annot in oneHot]
        rast = [False] + rast + [False]
        start = [i+1 for i,(a,b) in enumerate(zip(rast[1:],rast[:-1])) if (a and not b)]
        stop  = [i for i,(a,b) in enumerate(zip(rast[:-1],rast[1:])) if (a and not b)]
        bouts[name]['start'] = start
        bouts[name]['stop'] = stop
    return bouts


def bouts_to_rast(channel, n_frames, names):
    rast = ['other']*n_frames
    for beh in names:
        if beh in channel.keys():
            for row in channel[beh]:
                rowFix = [min(row[0],n_frames), min(row[1],n_frames-1)]
                rast[rowFix[0]:rowFix[1]+1] = [beh]*(rowFix[1]-rowFix[0]+1)
    return rast



def merge_channels(channel_dict, use_channels, end_frame, target_behaviors = []):
    # for now, we'll just merge kept channels together, in order listed. this can cause behaviors happening in
    # earlier channels to be masked by other behaviors in later channels. Specify behaviors to keep in
    # target_behaviors if desired, otherwise it merges everything.
    behFlag = 0
    changed_behavior_list = ['other'] * end_frame
    if not use_channels:
        use_channels = channel_dict.keys()
    for ch in use_channels:
        if (ch in channel_dict):
            keep_behaviors = target_behaviors if not target_behaviors==[] else filter(lambda x: x != '', set(channel_dict[ch].keys()))
            chosen_behavior_list = bouts_to_rast(channel_dict[ch], end_frame, keep_behaviors)
            if not(behFlag):
                changed_behavior_list = [annotated_behavior if annotated_behavior in keep_behaviors else 'other' for annotated_behavior in
                                         chosen_behavior_list]
                behFlag = 1
            else:
                changed_behavior_list = [anno[0] if anno[1] not in keep_behaviors else anno[1] for anno in zip(changed_behavior_list,chosen_behavior_list)]
        else:
            print('Did not find a channel' + ch)
            exit()
    return changed_behavior_list


def dump_labels_bento(labels, filename, moviename='', framerate=30, beh_list = ['mount','attack','sniff'], gt=None):

    # Convert labels to behavior bouts
    bouts = rast_to_bouts(labels,beh_list)
    # Open the file you want to write to.
    fp = open(filename, 'wb')
    ch_list = ['classifier_output']
    if gt is not None:
        ch_list.append('ground_truth')
        gt_bouts = rast_to_bouts(gt,beh_list)

    #####################################################

    # Write the header.
    fp.write('Bento annotation file\n')
    fp.write('Movie file(s):\n{}\n\n'.format(moviename))
    fp.write('Stimulus name:\n')
    fp.write('Annotation start frame: 1\n')
    fp.write('Annotation stop frame: {}\n'.format(len(labels)))
    fp.write('Annotation framerate: {}\n\n'.format(framerate))

    fp.write('List of channels:\n')
    fp.write('\n'.join(ch_list))
    fp.write('\n\n')

    fp.write('List of annotations:\n')
    fp.write('\n'.join(beh_list))
    fp.write('\n\n')

    #####################################################
    fp.write('{}----------\n'.format(ch_list[0]))
    for beh in beh_list:
        if beh in bouts.keys():
            fp.write('>{}\n'.format(beh))
            fp.write('Start\tStop\tDuration\n')
            for start,stop in zip(bouts[beh]['start'],bouts[beh]['stop']):
                fp.write('{}\t{}\t{}\t\n'.format(start,stop,stop-start+1))
            fp.write('\n')
    fp.write('\n')

    if gt is not None:
        fp.write('{}----------\n'.format(ch_list[1]))
        for beh in beh_list:
            if beh in gt_bouts.keys():
                fp.write('>{}\n'.format(beh))
                fp.write('Start\tStop\tDuration\n')
                for start,stop in zip(gt_bouts[beh]['start'],gt_bouts[beh]['stop']):
                    fp.write('{}\t{}\t{}\t\n'.format(start,stop,stop-start+1))
                fp.write('\n')
        fp.write('\n')

    fp.close()
    return
