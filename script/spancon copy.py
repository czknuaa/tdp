import spconv.pytorch as spconv
from torch import nn
import torch
import numpy as np
# class ExampleNet(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.net = spconv.SparseSequential(
#             spconv.SparseConv3d(32, 64, 3),  # 稀疏卷积
#             nn.BatchNorm1d(64),  # BatchNorm 层
#             nn.ReLU(),
#             spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),  # 子流形卷积
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             spconv.SparseConvTranspose3d(64, 64, 3, 2),  # 上采样
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             spconv.ToDense(),  # 转换为密集张量
#             nn.Conv3d(64, 64, 3),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
#         self.shape = shape

#     def forward(self, features, coors, batch_size):
#         coors = coors.int()  # 坐标必须是整数
#         x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
#         return self.net(x)
    
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3, 2, indice_key="cp0"),
            spconv.SparseInverseConv3d(64, 32, 3, indice_key="cp0"), # need provide kernel size to create weight
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)



from spconv.pytorch.utils import PointToVoxel
gen = PointToVoxel(
    vsize_xyz=[0.1, 0.1, 0.1], 
    coors_range_xyz=[-80, -80, -2, 80, 80, 6], 
    num_point_features=3, 
    max_num_voxels=5000, 
    max_num_points_per_voxel=5
)
pc = np.random.uniform(-10, 10, size=[1000, 3])
pc_th = torch.from_numpy(pc)
voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=True)
