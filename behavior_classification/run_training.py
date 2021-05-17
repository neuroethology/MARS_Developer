import os, sys
import MARS_train_test as mars
import argparse, pdb

parser = argparse.ArgumentParser()


parser.add_argument('--earlystopping', dest='earlystopping', default=10, type=int,
                    help='number of early stopping steps (default: 10)')

parser.add_argument('--maxdepth', dest='max_depth', default=3, type=int,
                    help='max tree depth (default: 3)')
                    
parser.add_argument('--minchild', dest='minchild', default=1, type=int,
                    help='min_child_weight (default: 1)')
                    
parser.add_argument('--dowavelet', dest='do_cwt', action='store_true',
                    default=False,
                    help='use wavelet transform on features (default: false)')
                    
parser.add_argument('--testonly', dest='testonly', action='store_true',
                    default=False,
                    help='skip classifier training (default: false)')

parser.add_argument('--behavior', dest='behavior', default='attack',
                    help = 'behavior to train (default: attack)')

parser.add_argument('--clf', dest='clf_pth', default='',
                    help = 'path to a specific classifier to test (otherwise will determine from other arguments)')

parser.add_argument('--videos', dest='vid_pth', default='/groups/Andersonlab/CMS273/',
                    help = 'path to directory containing folders TRAIN_ORIG and TEST_ORIG with training/test data (default: current directory)')

parser.add_argument('--suffix', dest='suffix', default='clf',
                    help='identifying suffix to append to clf name')

parser.add_argument('--trainset', dest='train', default='TRAIN',
                    help='training set to use (default TRAIN)')

parser.add_argument('--evalset', dest='val', default='EVAL',
                    help='validataion set to use (default EVAL)')

parser.add_argument('--testset', dest='test', default='TEST',
                    help='test set to use (default TEST)')

args = parser.parse_args()

behs = mars.get_beh_dict(args.behavior)

# tell the script where our training and test sets are located.
video_path = args.vid_pth
train_videos = [os.path.join(f, v) for f in args.train.split('-') for v in os.listdir(video_path + f)]
eval_videos = [os.path.join(f, v) for f in args.val.split('-') for v in os.listdir(video_path + f)]
test_videos = [os.path.join(f, v) for f in args.test.split('-') for v in os.listdir(video_path + f)]


# these are the parameters that define our classifier.
do_wnd = True if not args.do_cwt else False
clf_params = dict(clf_type='xgb', feat_type='top', do_cwt=args.do_cwt, do_wnd=do_wnd, 
                  early_stopping=args.earlystopping, max_depth=args.max_depth,
                  min_child_weight=args.minchild, clf_path_hardcoded=args.clf_pth,
                  downsample_rate=6, user_suff=args.suffix)

if not args.testonly:
    mars.train_classifier(behs, video_path, train_videos, eval_videos, clf_params=clf_params, verbose=1)

mars.test_classifier(behs, video_path, test_videos, clf_params=clf_params, verbose=1, doPRC=1)
