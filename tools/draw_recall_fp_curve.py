import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from ipdb import set_trace
import numpy as np

sns.set_theme(style="darkgrid")

path = './evaluation_results/20220509-124114-test_fp.pkl'
with open(path, 'rb') as f:
    eval_data = pkl.load(f)

fppi = eval_data['IoU@0.5/type_Pedestrian/range_OVERALL/length_[0, 4]']

recall = np.arange(101) / 100


sns.lineplot(recall, fppi)
plt.savefig('test.png')