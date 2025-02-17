import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import pdb

import open3d as o3d
import time

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class ConvEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim//2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(output_dim//2, output_dim, kernel_size=1, stride=1)
        
    def forward(self, x):
        # x shape: (B, N, 3)
        B, N, _ = x.shape
        
        # Reshape x for 1D convolution (B, 3, N)
        x = x.permute(0, 2, 1)  # x shape: (B, 3, N)
        
        # Apply Conv1d layers
        x = F.relu(self.conv1(x))  # x shape: (B, output_dim, N)
        x = F.relu(self.conv2(x))  # x shape: (B, output_dim, N)
        
        # Reshape back to (B, N, output_dim)
        x = x.permute(0, 2, 1)  # x shape: (B, N, output_dim)
        
        return x

# Transformer-based Model for Point Cloud
class PointCloudTransformer(nn.Module):
    def __init__(self, input_dim=3, embeding_dim =192,output_dim=128, n_heads=4, num_layers=4,dff = 512):
        super(PointCloudTransformer, self).__init__()
        
        # 1D Convolutional Embedding Layer
        self.embedding = ConvEmbedding(input_dim, embeding_dim)
        
        # Transformer Layer (using nn.TransformerEncoder)
   
        encoder_layer = nn.TransformerEncoderLayer(d_model=embeding_dim, nhead=n_heads, dim_feedforward= dff, batch_first=True,dropout = 0.1)
       
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.convout1 = nn.Conv1d(embeding_dim, output_dim//2, kernel_size=1, stride=1)
        self.convout2 = nn.Conv1d(output_dim//2, output_dim, kernel_size=1, stride=1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(output_dim//2, output_dim//2),  # 第一层线性变换
        #     nn.ReLU(),                            # 激活函数
        #     nn.Linear(output_dim//2, output_dim)  # 输出为新的维度
        # )
    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        
       # print(x.shape)
        # Embedding step (1D Conv)
        x = self.embedding(x)  # x: (B, N, output_dim)

        # Reshape for transformer (Seq, Batch, Feature)
        # x = x.permute(1, 0, 2)  # x: (N, B, output_dim)
        
        # Pass through Transformer encoder
        x = self.transformer(x)  # x: (N, B, output_dim)
        # x = x.permute(0, 2, 1) 
        # x = F.relu(self.convout1(x))
        # x = F.relu(self.convout2(x))
        # x = x.permute(0, 2, 1) 
        # Apply average pooling (B, output_dim)
        # x = x.permute(1, 0, 2)  # x: (B, N, output_dim)
        x,_ = x.max(dim=1)  # x: (B, output_dim)
        # x = self.mlp(x)
        
        return x

def Vis_PC(xyz_filtered):
    # 创建 Open3D 点云对象
    pcd_clustered = o3d.geometry.PointCloud()
    pcd_clustered.points = o3d.utility.Vector3dVector(xyz_filtered)
   # pcd_clustered.colors = o3d.utility.Vector3dVector(colored_labels)
    stime = time.time()
    voxel_size = 0.05  # 体素大小
    downpcd = pcd_clustered.voxel_down_sample(voxel_size)
    print(time.time() - time.time())
    # 可视化聚类后的点云
    o3d.visualization.draw_geometries([downpcd], window_name="DBSCAN Clustering")
    num_points = len(downpcd.points)  # 或者 num_points = np.asarray(downpcd.points).shape[0]


    
class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
              #  self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
               # self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        ###
        self.extractor = PointCloudTransformer()

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
    
        # 生成掩码 (B, N)
       
        pn_feat = self.extractor(points)    # B * out_channel
     
        # state = observations[self.state_key]
        # state_feat = self.state_mlp(state)  # B * 64
        final_feat = pn_feat
        # final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels