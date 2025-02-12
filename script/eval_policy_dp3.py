import sys
sys.path.insert(0, './policy/3D-Diffusion-Policy/3D-Diffusion-Policy')
sys.path.append('./')

import torch  
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
import hydra
import pathlib

from dp3_policy import *

import yaml
from datetime import datetime
import importlib

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def class_decorator(task_name):
    #动态导入模块  相当于 import env.taskname
    envs_module = importlib.import_module(f'envs.{task_name}')    
    try:
        #从导入的模块中获取类
        env_class = getattr(envs_module, task_name)
        #实例化类 env_instance 为实例化的对象
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    #返回实例化的对象
    return env_instance


# 获取摄像头配置函数
# 输入：camera_tpye 摄像头的类型（D435 or L515）
# 返回：对应摄像头类型的摄像头参数
def get_camera_config(camera_type):
    #读取文件路径
    camera_config_path = os.path.join(parent_directory, '../task_config/_camera_config.yml')
    
    #检查路进是否存在
    assert os.path.isfile(camera_config_path), "task config file is missing"
    
    #加载参数
    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    #检查对应的摄像头型号是否存在
    assert camera_type in args, f'camera {camera_type} is not defined'

    #返回对应型号的参数
    return args[camera_type]



#加载模型
def load_model(model_path):
    model = torch.load(model_path)
    #将模型设为eval模式
    model.eval() 
    return model

TASK = None


#使用hydra 实例化Dataset类和RobotRunner类  运行到main时开始执行
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        '../../policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d', 'config'))
)

def main(cfg):
    #从刚刚导入的cfg中导入参数
    global TASK
    TASK = cfg.task.name
    print('Task name:', TASK)
    checkpoint_num = cfg.checkpoint_num
    expert_data_num = cfg.expert_data_num
    seed = cfg.training.seed
    head_camera_type = cfg.head_camera_type

    with open(f'./task_config/{cfg.raw_task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['head_camera_type'] = head_camera_type
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']
    head_camera_config = 'fovy' + str(args['head_camera_fovy']) + '_w' + str(args['head_camera_w']) + '_h' + str(args['head_camera_h'])
    
    wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    args['wrist_camera_w'] = wrist_camera_config['w']
    args['wrist_camera_h'] = wrist_camera_config['h']
    wrist_camera_config = 'fovy' + str(args['wrist_camera_fovy']) + '_w' + str(args['wrist_camera_w']) + '_h' + str(args['wrist_camera_h'])

    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']
    front_camera_config = 'fovy' + str(args['front_camera_fovy']) + '_w' + str(args['front_camera_w']) + '_h' + str(args['front_camera_h'])

    # output camera config
    print('============= Camera Config =============\n')
    print('Head Camera Config:\n    type: '+ str(args['head_camera_type']) + '\n    fovy: ' + str(args['head_camera_fovy']) + '\n    camera_w: ' + str(args['head_camera_w']) + '\n    camera_h: ' + str(args['head_camera_h']))
    print('Wrist Camera Config:\n    type: '+ str(args['wrist_camera_type']) + '\n    fovy: ' + str(args['wrist_camera_fovy']) + '\n    camera_w: ' + str(args['wrist_camera_w']) + '\n    camera_h: ' + str(args['wrist_camera_h']))
    print('Front Camera Config:\n    type: '+ str(args['front_camera_type']) + '\n    fovy: ' + str(args['front_camera_fovy']) + '\n    camera_w: ' + str(args['front_camera_w']) + '\n    camera_h: ' + str(args['front_camera_h']))
    print('\n=======================================')

    args['expert_seed'] = seed
    args['expert_data_num'] = expert_data_num
    
    #实例化任务类
    task = class_decorator(args['task_name'])

    st_seed = 100000 * (1+seed)       #随机数种子
    suc_nums = []                     #用于储存成功的此书
    test_num = 100                    #评估总次数，默认100次
    topk = 1                          #成功率
    
    dp3 = DP3(cfg, checkpoint_num)    #实例化DP3模型

    #评估模型
    st_seed, suc_num = test_policy(task, args, dp3, st_seed, test_num=test_num)
    suc_nums.append(suc_num)
    
    #计算top k成功率并保存评估结果
    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    if not cfg.policy.use_pc_color: # result_dp3/task_camera/expert_data_num/ckpt_seed
        save_dir  = f'eval_result/dp3/{TASK}_{head_camera_type}/{expert_data_num}'
    else:
        save_dir = f'eval_result/dp3_w_rgb/{TASK}_{head_camera_type}/{expert_data_num}'
    
    file_path = os.path.join(save_dir, f'ckpt_{checkpoint_num}_seed_{seed}.txt')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, 'w') as file:
        file.write(f'Timestamp: {current_time}\n\n')

        file.write(f'Checkpoint Num: {checkpoint_num}\n')
        
        file.write('Successful Rate of Diffenent checkpoints:\n')
        file.write('\n'.join(map(str, np.array(suc_nums) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate (every):\n')
        file.write('\n'.join(map(str, np.array(topk_success_rate) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate:\n')
        file.write(f'\n'.join(map(str, np.array(topk_success_rate) / (topk * test_num))))
        file.write('\n\n')

    print(f'Data has been saved to {file_path}')
    



#核心函数，用于评估策略网络
def test_policy(Demo_class, args, dp3, st_seed, test_num=20):
    ##初始化相关参数
    global TASK
    epid = 0      
    seed_list=[]  
    suc_num = 0   
    expert_check = True
    print("Task name: ",args["task_name"])


    Demo_class.suc = 0
    Demo_class.test_num =0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    

    now_seed = st_seed

    #执行测试循环  直到成功次数达到test——num，任务结束
    while succ_seed < test_num:

        #设置渲染频率
        render_freq = args['render_freq']
        args['render_freq'] = 0
        
        #如果执行专家检查
        if expert_check:
            try:
                #初始化任务
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
                #执行一次演示
                Demo_class.play_once()
                #关闭任务
                Demo_class.close()
            #报错的话打印错误信息
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 1
                args['render_freq'] = render_freq
                print('error occurs !')
                continue
        
        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args['render_freq'] = render_freq
            continue

        args['render_freq'] = render_freq

        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        Demo_class.apply_dp3(dp3, args)

        now_id += 1
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()
        dp3.env_runner.reset_obs()
        print(f"{TASK} success rate: {Demo_class.suc}/{Demo_class.test_num}, current seed: {now_seed}\n")
        Demo_class._take_picture()
        now_seed += 1

    return now_seed, Demo_class.suc

if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    main()
