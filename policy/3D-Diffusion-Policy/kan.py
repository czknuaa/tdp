import zarr
import numpy as np
import open3d as o3d
import time
import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time


import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle
import torch
import torch
import torch.nn.functional as F
import torch
def voxel_downsample_gpu(points_np, voxel_size):
    """
    使用体素网格下采样点云。
    
    参数：
    points (Tensor): 输入点云数据，大小为 N x 3，其中 N 是点的数量。
    voxel_size (float): 体素网格的大小。

    返回：
    Tensor: 下采样后的点云数据。
    """
    points = torch.from_numpy(points_np)
    points.cuda()
    # 计算每个点所在的体素位置（将坐标除以体素大小并取整）
    voxel_indices = torch.floor(points / voxel_size)
    
    # 去除重复的体素索引，得到唯一的体素
    _, unique_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
    
    # 返回每个体素的代表点，这里选择每个体素内的第一个点
    downsampled_points = points[torch.unique(unique_indices)]
    
    return downsampled_points

down_spamle_mod =2
def Voxel_Down_Sample_without_tran(pcd,voxel_size):
  
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    num_points = len(downsampled_pcd.points)
    print(f"Number of points after downsampling: {num_points}")


    # 如果需要，将点云的点和颜色拼接在一起
   # pcd_with_colors = np.hstack((points, colors))  # shape: (N, 6)，包含点和颜色信息  
  
    return downsampled_pcd

def voxel_downsample(points, voxel_size):
   
    voxel_grid = np.floor(points / voxel_size)
    
    # Step 2: 获取唯一的体素位置，选择每个体素的代表点（此处为第一个点）
    unique_voxels, indices = np.unique(voxel_grid, axis=0, return_index=True)
    
    # Step 3: 选择每个体素内的第一个点作为代表点
    downsampled_points = points[indices]
    print("num point",downsampled_points.shape)
    
    return downsampled_points

import numpy as np

def random_uniform_sampling_point_cloud(point_cloud, num_samples):
    """
    对点云数据进行随机均匀采样
    
    参数:
        point_cloud (numpy.ndarray): 原始点云数据，形状为 (N, 3)，N 为点的数量
        num_samples (int): 需要采样的点的数量
    
    返回:
        numpy.ndarray: 采样后的点云，形状为 (num_samples, 3)
    """
    # 从点云中随机选择 num_samples 个点
    random_indices = np.random.choice(point_cloud.shape[0], num_samples, replace=False)
    sampled_points = point_cloud[random_indices]
    return sampled_points



def Voxel_Down_Sample(xyz,voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
  
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    num_points = len(downsampled_pcd.points)
    print(f"Number of points after downsampling: {num_points}")

    points = np.asarray(downsampled_pcd.points)  # 转换为 numpy 数组
 
    # 如果需要，将点云的点和颜色拼接在一起
   # pcd_with_colors = np.hstack((points, colors))  # shape: (N, 6)，包含点和颜色信息  
  
    return points

import numpy as np





def get_points(file_path,episode_id_,frame_id_):
    # 1️⃣ 读取 Zarr 文件
    zarr_file = file_path # 确保路径正确
    #zarr_file = os.path.join("data", "empty_cup_place_L515_50.zarr")
    print(zarr_file)
    zarr_root = zarr.open(zarr_file, mode="r")

    # 2️⃣ 获取点云数据和元数据
    point_cloud_data = zarr_root["data/point_cloud"][:]  # 形状: (21095, 1024, 6)
    episode_ends = zarr_root["meta/episode_ends"][:]  # 形状: (100,)

    # 3️⃣ 按 `episode_ends` 划分点云
    episodes = []
    start_idx = 0  # 每个 episode 的起始索引
    for end_idx in episode_ends:
        episodes.append(point_cloud_data[start_idx:end_idx])  # 切片每个 episode
        start_idx = end_idx  # 下一个 episode 的开始索引

    # 4️⃣ 选择某个演示 Episode 进行可视化
    episode_id = episode_id_  # 选择第 20 个演示
    frame_id = frame_id_  # 选择该演示的第 5 帧

    if episode_id >= len(episodes):
        raise ValueError(f"Episode ID {episode_id} 超出范围，最大值为 {len(episodes)-1}")

    selected_episode = episodes[episode_id]
    if frame_id >= len(selected_episode):
        raise ValueError(f"Frame ID {frame_id} 超出范围，该演示共有 {len(selected_episode)} 帧")

    # 5️⃣ 获取点云数据
    frame_data = selected_episode[frame_id]  # 形状 (1024, 6)
    xyz = frame_data[:, :3]  # 提取坐标
    colors = frame_data[:, 3:6]  # 提取颜色

    # 颜色归一化
    if colors.max() > 1:
        colors /= 255.0
    return xyz,colors







file_path = "data/empty_cup_place_L515_50.zarr"
episode_id = 3
frame_id = 5
xyz,colors = get_points(file_path,episode_id,frame_id)

start_time  = time.time()
for i in range(100):
    if down_spamle_mod == 1:
         xyz_down = Voxel_Down_Sample(xyz,0.017)
    elif down_spamle_mod == 2:
        xyz_down = random_uniform_sampling_point_cloud(xyz,256)
print(time.time()-start_time)


# 6️⃣ 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_down)
pcd.colors = o3d.utility.Vector3dVector(colors)

# start_time  = time.time()
# for i in range(100):
#     pcd_down = Voxel_Down_Sample_without_tran(pcd,0.017)
# print(time.time()-start_time)
# # 7️⃣ 可视化该帧
o3d.visualization.draw_geometries([pcd], window_name=f"Episode {episode_id}, Frame {frame_id}")
