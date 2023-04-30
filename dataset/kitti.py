import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from utils import read_pickle, read_points, bbox_camera2lidar
from dataset import point_range_filter, data_augment, polar_stitch, rotate_sample


class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Kitti(Dataset):

    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }

    def __init__(self, data_root, split, pts_prefix='velodyne_reduced'):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        self.data_infos = read_pickle(os.path.join(data_root, f'kitti_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())
        db_infos = read_pickle(os.path.join(data_root, 'kitti_dbinfos_train.pkl'))
        db_infos = self.filter_db(db_infos)

        db_sampler = {}
        for cat_name in self.CLASSES:
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        self.data_aug_config=dict(
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10)
                ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ), 
            point_range_filter=[0, -39.68, -3, 69.12, 39.68, 1],
            object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1]             
        )

    def remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
        return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info = \
            data_info['image'], data_info['calib'], data_info['annos']
    
        # point cloud input
        velodyne_path = data_info['velodyne_path'].replace('velodyne', self.pts_prefix)
        pts_path = os.path.join(self.data_root, velodyne_path)
        pts = read_points(pts_path)
        
        # calib input: for bbox coordinates transformation between Camera and Lidar.
        # because
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        # annotations input
        annos_info = self.remove_dont_care(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_diffs = annos_info['difficulty']

        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
        gt_labels = np.array([self.CLASSES.get(name, -1) for name in annos_name])

        # polarmix 
        # info of data to mix in
        index2 = np.random.randint(len(self.sorted_ids))
        data2_info = self.data_infos[self.sorted_ids[index2]]
        calib2_info, annos2_info = data2_info['calib'], data2_info['annos']
        # points to mix in 
        velodyne2_path = data2_info['velodyne_path'].replace('velodyne', self.pts_prefix)
        pts2_path = os.path.join(self.data_root, velodyne2_path)
        pts2 = read_points(pts2_path)
        # calib info of data to mix in
        tr_velo_to_cam2 = calib2_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect2 = calib2_info['R0_rect'].astype(np.float32)
        # annotations 
        annos2_info = self.remove_dont_care(annos2_info)
        annos2_name = annos2_info['name']
        annos2_location = annos2_info['location']
        annos2_dimension = annos2_info['dimensions']
        rotation2_y = annos2_info['rotation_y']
        gt_diffs2 = annos2_info['difficulty']

        # gt info to mix in
        gt_bboxes2 = np.concatenate([annos2_location, annos2_dimension, rotation2_y[:, None]], axis=1).astype(np.float32)
        gt_bboxes_3d2 = bbox_camera2lidar(gt_bboxes2, tr_velo_to_cam2, r0_rect2)
        gt_labels2 = np.array([self.CLASSES.get(name, -1) for name in annos2_name])

        # polarmix swapping azymuth ranges
        theta1 = (np.random.random() - 1) * np.pi
        theta2 = theta1 + np.pi * 1
        # polarmix rotation omegas
        # omegas = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]
        omegas = [np.random.random() * np.pi * 2 / 3]
        # polarmix - swap
        # if np.random.random() < 0.5:
        #     pts, gt_bboxes_3d, gt_labels, annos_name, gt_diffs = polar_stitch(pts, 
        #                                                                     pts2, 
        #                                                                     gt_bboxes_3d, 
        #                                                                     gt_bboxes_3d2, 
        #                                                                     gt_labels, 
        #                                                                     gt_labels2, 
        #                                                                     annos_name, 
        #                                                                     annos2_name,
        #                                                                     gt_diffs, 
        #                                                                     gt_diffs2, 
        #                                                                     theta1, 
        #                                                                     theta2)
            
        # # polarmix - rotate addition
        # if np.random.random() < 1.0:
        #     pts_add, gt_bboxes_3d_add, gt_labels_add, annos_name_add, gt_diffs_add = rotate_sample(pts2, 
        #                                                                                            gt_bboxes_3d2, 
        #                                                                                            gt_labels2, 
        #                                                                                            annos2_name, 
        #                                                                                            gt_diffs2, 
        #                                                                                            omegas)
            
        #     pts = np.concatenate((pts, pts_add), axis=0)
        #     gt_bboxes_3d = np.concatenate((gt_bboxes_3d, gt_bboxes_3d_add), axis=0)
        #     gt_labels = np.concatenate((gt_labels, gt_labels_add), axis=0)
        #     annos_name = np.concatenate((annos_name, annos_name_add), axis=0)
        #     gt_diffs = np.concatenate((gt_diffs, gt_diffs_add), axis=0)

        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels, 
            'gt_names': annos_name,
            'difficulty': gt_diffs,
            'image_info': image_info,
            'calib_info': calib_info
        }


        if self.split in ['train', 'trainval']:
            data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)
        else:
            data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])

        return data_dict

    def __len__(self):
        return len(self.data_infos)
 

if __name__ == '__main__':
    
    kitti_data = Kitti(data_root='/mnt/ssd1/lifa_rdata/det/kitti', 
                       split='train')
    kitti_data.__getitem__(9)
