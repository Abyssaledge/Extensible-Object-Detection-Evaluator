## A simple and flexible object detection evaluator in COCO style

### Features 
- Extensible user interfaces to deal with different data formats.
- Support customized evaluation breakdowns, e.g., object size in COCO, difficulty in KITTI, velocity and range in Waymo.
- Interface for general matching scores, e.g. 2D IoU, 3D rotated IoU, center distance.

### Install
- `pip install treelib`
- Clone this repo and run `pip install .` in the cloned directory.
---
### Prepare predictions and groundtruth
You need to define a function to read the predictions and groundtruth. 
```
def read_prediction(path):
    ...
    return results
```
where the `results` is a dictionary using sample_id (image_id) as key and each item is a dictionary contains at least `box`, `score` and `type`:
```
# ndarray in shape of [N, C], where N is the number of bboxes in this sample and C is the box dimension
boxes = ... 

# ndarray in shape [N,]
scores = ... 

# ndarray in shape of [N,]
types = ... 

results[sample_id] = dict(box=boxes, score=scores, type=types)
```
And you need to define a another function to read groundtruth as the same way. The items of returned dict contains at list `box` and `type`.

---
### Custimize matching score function
If you are going to evaluate a 2D detector, you may want to define a 2D IoU function, which will be automatically used in the matching process. 

```
def iou_2d(box1, box2):
    # box1 in shape of [N1, 4], which is the box item defined above.
    # box2 in shape of [N2, 4], which is the box item defined above.

    # a iou matrix in shape of [N1, N2]
    iou_matrix = ...
    return iou_matrix
```

---
### Customize breakdowns if you like
