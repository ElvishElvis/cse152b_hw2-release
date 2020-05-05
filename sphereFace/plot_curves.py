
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from tensorboardX import SummaryWriter
from torchvision import datasets

from utils.utils import getWriterPath


def tb_scalar_dict(writer, scalar_dict, task='training'):
    for element in list(scalar_dict):
        obj = scalar_dict[element]
        for i in range(obj.shape[0]):
            writer.add_scalar(task + '-' + element, obj[i], i)

# resnet18 = models.resnet18(False)
# writer = SummaryWriter()
# exp = 'add_bn'
exps = ['checkpoint', 'add_bn_3', '../cosFace/cosf_sph20_2']
print("exp: ", exps)
scalar_dict = {}

for exp in exps:
    writer = SummaryWriter(getWriterPath(task=exp, date=True))
    print("exp: ", exp)
    scalar_dict['loss'] = np.load(exp+'/loss.npy')
    scalar_dict['accuracy'] = np.load(exp+'/accuracy.npy')
    tb_scalar_dict(writer, scalar_dict)




