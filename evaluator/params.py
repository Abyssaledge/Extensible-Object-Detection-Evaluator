from abc import ABC, abstractmethod
import numpy as np
from ipdb import set_trace

def waymo_range_breakdown(o):
    return np.linalg.norm(o['box'][:, :3], ord=2, axis=1)

def waymo_length_breakdown(o):
    return o['box'][:, 4]

class BaseParam(ABC):
    def __init__(self, pd_path, gt_path, interval=1, update_sep=None, update_insep=None):
        self.pd_path = pd_path
        self.gt_path = gt_path

        self.add_breakdowns()
        self.add_iou_function()
        self.add_input_function()

        if update_sep is not None:
            self.separable_breakdowns.update(update_sep)
        if update_insep is not None:
            self.inseparable_breakdowns.update(update_insep)

        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)

        self.sampling_interval = interval
    
    @abstractmethod
    def add_breakdowns(self):
        pass

    @abstractmethod
    def add_iou_function(self):
        pass

    @abstractmethod
    def add_input_function(self):
        pass

class WaymoBaseParam(BaseParam):

    def __init__(self, pd_path, gt_path, interval=1, update_sep=None, update_insep=None):
        super().__init__(pd_path, gt_path, interval, update_sep, update_insep)
        self.iouThrs = [0.7, 0.5]
    
    def add_breakdowns(self):

        self.separable_breakdowns = {
            'type':('Vehicle', 'Pedestrian', 'Cyclist'), 
            'range':([0, 30], [30, 50], [50, 80], None), # None means the union of all ranges
        }
        self.breakdown_func_dict = {'range': waymo_range_breakdown}

        self.inseparable_breakdowns = {}
    
    def add_iou_function(self):
        from evaluator.ious import get_waymo_iou_matrix
        self.iou_calculate_func = get_waymo_iou_matrix

    def add_input_function(self):
        from evaluator.utils import get_waymo_object
        self.read_prediction_func = get_waymo_object
        self.read_groundtruth_func = get_waymo_object


class WaymoLengthParam(WaymoBaseParam):

    def __init__(self, pd_path, gt_path, length_range, interval=1, update_sep=None, update_insep=None):
        super().__init__(pd_path, gt_path, interval, update_sep, update_insep)
        self.inseparable_breakdowns['length'] = length_range
        self.breakdown_func_dict['length'] = waymo_length_breakdown



