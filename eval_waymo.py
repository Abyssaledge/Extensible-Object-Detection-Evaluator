# evaluate the waymo xx.bin files in COCO-like protocal
from evaluator.eval import Evaluator
import argparse

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--pd-path', type=str, default='/mnt/truenas/scratch/lve.fan/transdet3d/work_dirs/rsn_vote_4block_full_lrx05/results.bin')
# parser.add_argument('--pd-path', type=str, default='/mnt/truenas/scratch/lve.fan/transdet3d/gt.bin')
parser.add_argument('--gt-path', type=str, default='/mnt/truenas/scratch/lve.fan/transdet3d/gt.bin')
parser.add_argument('--save-folder', type=str, default='')
parser.add_argument('--save-suffix', type=str, default='')
parser.add_argument('--interval', type=int, default=10)
# process
args = parser.parse_args()


if __name__ == '__main__':
    pd_path = args.pd_path
    # pd_path = ''
    gt_path = args.gt_path 

    from evaluator.params import WaymoBaseParam, WaymoLengthParam
    # update = {'type':('Vehicle', 'Ped')}
    update_sep = {'type':('Pedestrian',)}
    # update_sep = None

    params = WaymoLengthParam(pd_path, gt_path, [None, [0, 4], [4, 8], [8, 20]], interval=args.interval, update_sep=update_sep)
    # params = WaymoBaseParam(pd_path, gt_path, [None, [0, 4], [4, 8], [8, 20]], interval=args.interval, update_sep=update)
    params.save_suffix = args.save_suffix
    params.iouThrs = [0.05,]

    # params.save_folder = '/mnt/truenas/scratch/lve.fan/UniDetEval/results'

    evaluator = Evaluator(params, debug=False)
    evaluator.run()