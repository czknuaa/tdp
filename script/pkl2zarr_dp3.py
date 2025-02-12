import pdb, pickle, os
import numpy as np
import open3d as o3d
from copy import deepcopy
import zarr, shutil
import argparse

def main():

    #接受并解析传递的参数
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_name', type=str)
    parser.add_argument('head_camera_type', type=str)
    parser.add_argument('expert_data_num', type=int)

    args = parser.parse_args()
    
    #传递参数
    #可视化数据 = false
    visualize_pcd = False

    task_name = args.task_name   #任务名称
    num = args.expert_data_num   #数据数量
    current_ep, num = 0, num     #current_ep 当前处理的数据索引
    head_camera_type = args.head_camera_type   
    load_dir = f'./data/{task_name}_{head_camera_type}_pkl'
    
    total_count = 0    #数据总数
    
    #zarr的保存路径
    save_dir = f'./policy/3D-Diffusion-Policy/data/{task_name}_{head_camera_type}_{num}.zarr'
   
    #如果路径存在，则删除原来的文件
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    point_cloud_arrays, episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], [], []
    


    #循环条件： epsiode n 存在， 处理的数据数量<总的需要处理的数据数量
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0        #每个epsiode中plk文件的编号索引
        point_cloud_sub_arrays = [] #点云子数组
        state_sub_arrays = []
        action_sub_arrays = [] 
        joint_action_sub_arrays = []
        episode_ends_sub_arrays = []
        
        #循环条件 episode n 中的 plk i 文件存在
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            #打开pkl文件，并加载道data中
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            #读取pkl中的数据
            pcd = data['pointcloud'][:,:]
            action = data['endpose']
            

            joint_action = data['joint_action']

            point_cloud_sub_arrays.append(pcd)
            state_sub_arrays.append(joint_action)
            action_sub_arrays.append(action)
            joint_action_sub_arrays.append(joint_action)

            if visualize_pcd:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data['pcd']['points'])
                pcd.colors = o3d.utility.Vector3dVector(data['pcd']['colors'])
                o3d.visualization.draw_geometries([pcd])

            file_num += 1        #pkl ++
            total_count += 1     #总文件数 ++
             
        current_ep += 1 #episode 编号++
 
        episode_ends_arrays.append(deepcopy(total_count))    #用于划分数据集中的数据属于哪个episode
        point_cloud_arrays.extend(point_cloud_sub_arrays)
        action_arrays.extend(action_sub_arrays)
        state_arrays.extend(state_sub_arrays)
        joint_action_arrays.extend(joint_action_sub_arrays)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    point_cloud_arrays = np.array(point_cloud_arrays)
    joint_action_arrays = np.array(joint_action_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)   #clevel 为压缩等级   shuffle为是否打乱
    action_chunk_size = (100, action_arrays.shape[1])    #设置chunking大小，第二个变量定义了哪个维度不分块
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])


    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()
