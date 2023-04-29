import copy
import numba
import numpy as np
import os
import pdb
from pcdet.utils.box_utils import boxes_to_corners_3d, remove_points_in_boxes3d
from utils import bbox3d2bevcorners, box_collision_test, read_points, \
    remove_pts_in_bboxes, limit_period

# swapping for polarmix
def polar_stitch(pts1, pts2, gt_bboxes_3d1, gt_bboxes_3d2, gt_labels1, gt_labels2, gt_names1, gt_names2, gt_diffs1, gt_diffs2, theta1, theta2):
    pt_yaw1 = np.arctan2(pts1[:, 1], pts1[:, 0])
    pt_yaw2 = np.arctan2(pts2[:, 1], pts2[:, 0])

    pt_idx1 = np.where((pt_yaw1 > theta1) & (pt_yaw1 < theta2))
    pt_idx2 = np.where((pt_yaw2 > theta1) & (pt_yaw2 < theta2))

    pts1_new = np.delete(pts1, pt_idx1, axis=0)
    pts1_new = np.concatenate((pts1_new, pts2[pt_idx2]))

    bbox_yaw1 = np.arctan2(gt_bboxes_3d1[:, 1], gt_bboxes_3d1[:, 0])
    bbox_yaw2 = np.arctan2(gt_bboxes_3d2[:, 1], gt_bboxes_3d2[:, 0])

    bbox_idx1 = np.where((bbox_yaw1 > theta1) & (bbox_yaw1 < theta2))
    bbox_idx2 = np.where((bbox_yaw2 > theta1) & (bbox_yaw2 < theta2))

    gt_bboxes_3d1_new = np.delete(gt_bboxes_3d1, bbox_idx1, axis=0)
    gt_bboxes_3d1_new = np.concatenate((gt_bboxes_3d1_new, gt_bboxes_3d2[bbox_idx2]))

    gt_labels1_new = np.delete(gt_labels1, bbox_idx1, axis=0)
    gt_labels1_new = np.concatenate((gt_labels1_new, gt_labels2[bbox_idx2]))

    gt_names1_new = np.delete(gt_names1, bbox_idx1, axis=0)
    gt_names1_new = np.concatenate((gt_names1_new, gt_names2[bbox_idx2]))

    gt_diffs1_new = np.delete(gt_diffs1, bbox_idx1, axis=0)
    gt_diffs1_new = np.concatenate((gt_diffs1_new, gt_diffs2[bbox_idx2]))

    if(len(pts1_new) > 0 and len(gt_labels1_new) > 0):
        return pts1_new, gt_bboxes_3d1_new, gt_labels1_new, gt_names1_new, gt_diffs1_new
    else:
        return pts1, gt_bboxes_3d1, gt_labels1, gt_names1, gt_diffs1

# rotating for polarmix
def rotate_sample(pts, gt_bboxes_3d, gt_labels, annos_name, gt_diffs, omegas):
    pts_add = [pts]
    gt_bboxes_3d_add = [gt_bboxes_3d]
    labels_add = [gt_labels]
    names_add = [annos_name]
    diffs_add = [gt_diffs]

    for om in omegas:
        mat = np.array([[np.cos(om), np.sin(om), 0], [-np.sin(om), np.cos(om), 0], [0, 0, 1]])

        pts_new = np.zeros_like(pts)
        pts_new[:, :3] = pts[:, :3] @ mat
        pts_new[:, 3] = pts[:, 3]

        gt_bboxes_3d_new = np.zeros_like(gt_bboxes_3d)
        gt_bboxes_3d_new[:, :3] = gt_bboxes_3d[:, :3] @ mat
        gt_bboxes_3d_new[:, 3:] = gt_bboxes_3d[:, 3:]

        pts_add.append(pts_new)
        gt_bboxes_3d_add.append(gt_bboxes_3d_new)
        labels_add.append(gt_labels)
        names_add.append(annos_name)
        diffs_add.append(gt_diffs)

    return np.concatenate(pts_add, axis=0), np.concatenate(gt_bboxes_3d_add, axis=0), np.concatenate(labels_add, axis=0), np.concatenate(names_add, axis=0), np.concatenate(diffs_add, axis=0)


# shape aware dropout
def octodron_dropout(data_dict, human_more_dropout=False, car_more_dropout=True):
    pts, gt_bboxes_3d, gt_labels  = data_dict['pts'], data_dict['gt_bboxes_3d'], data_dict['gt_labels']

    gt_bbox3d_corner_pts = boxes_to_corners_3d(gt_bboxes_3d)

    # randomly pick for each gt bbox one of the 8 octodrons to dropout
    idx_octodron = np.random.randint(8, size=gt_bbox3d_corner_pts.shape[0])
    corners_picked = gt_bbox3d_corner_pts[np.arange(gt_bbox3d_corner_pts.shape[0]), idx_octodron]

    # construct the new centroids and new l, w, h dims of the filtering 3d bboxes
    # the yaw around z axis remains unchanged
    new_centers = (gt_bboxes_3d[:, :3] + corners_picked) / 2
    new_dims = gt_bboxes_3d[:, 3:-1] / 2
    
    filter_bboxes = np.hstack((new_centers, new_dims, gt_bboxes_3d[:, -1:]))

    # if dropping out one more octodron adjacent to the already chose ones for human objects (pedestrians/cyclists)
    # add additional filtering bboxes only for human objects
    if human_more_dropout or car_more_dropout:
        # get the indices for the gt bboxes corresponding to human objects
        # idx_human = np.where((gt_labels == 0) | (gt_labels == 1))
        # idx_human = np.where((gt_labels == 2))
        if human_more_dropout and not car_more_dropout:
            idx_additional = np.where((gt_labels == 0) | (gt_labels == 1))
        elif car_more_dropout and not human_more_dropout:
            idx_additional = np.where((gt_labels == 2))
        else:
            idx_additional = np.where((gt_labels == 0) | (gt_labels == 1) | (gt_labels == 2))
        # shift the original choice for the octodron chosen for these objects by 1 to construct additional choices
        # wrap at 8
        idx_additional_octodron = (idx_octodron[idx_additional] + 1) % 8

        # store only those gt bboxes for human objects and the corresponding corner points in new variables
        gt_bboxes_additional = gt_bboxes_3d[idx_additional]
        gt_bboxes_additional_corner_pts = gt_bbox3d_corner_pts[idx_additional]

        # perform picking of additional corner points (corresponding to the additional octodrons)
        additional_corners_picked = gt_bboxes_additional_corner_pts[np.arange(gt_bboxes_additional_corner_pts.shape[0]), idx_additional_octodron]

        # construct the additional filtering bboxes for these human objects - center points, l, w, h dims
        # the yaw around z axis remains unchanged
        additional_new_centers = (gt_bboxes_additional[:, :3] + additional_corners_picked) / 2
        additional_new_dims = gt_bboxes_additional[:, 3:-1] / 2
        additional_filter_bboxes = np.hstack((additional_new_centers, additional_new_dims, gt_bboxes_additional[:, -1:]))

        # vstack these additional filtering bboxes to the existing ones
        filter_bboxes = np.vstack((filter_bboxes, additional_filter_bboxes))

    # filter out points in these filtering bboxes
    data_dict['pts'] = remove_points_in_boxes3d(pts, filter_bboxes)

    return data_dict

def dbsample(CLASSES, data_root, data_dict, db_sampler, sample_groups):
    '''
    CLASSES: dict(Pedestrian=0, Cyclist=1, Car=2)
    data_root: str, data root
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    db_infos: dict(Pedestrian, Cyclist, Car, ...)
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
    gt_difficulty = data_dict['difficulty']
    image_info, calib_info = data_dict['image_info'], data_dict['calib_info']

    sampled_pts, sampled_names, sampled_labels = [], [], []
    sampled_bboxes, sampled_difficulty = [], []

    avoid_coll_boxes = copy.deepcopy(gt_bboxes_3d)
    for name, v in sample_groups.items():
        # 1. calculate sample numbers
        sampled_num = v - np.sum(gt_names == name)
        if sampled_num <= 0:
            continue

        # 2. sample databases bboxes
        sampled_cls_list = db_sampler[name].sample(sampled_num)
        sampled_cls_bboxes = np.array([item['box3d_lidar'] for item in sampled_cls_list], dtype=np.float32)

        # 3. box_collision_test
        avoid_coll_boxes_bv_corners = bbox3d2bevcorners(avoid_coll_boxes)
        sampled_cls_bboxes_bv_corners = bbox3d2bevcorners(sampled_cls_bboxes)
        coll_query_matrix = np.concatenate([avoid_coll_boxes_bv_corners, sampled_cls_bboxes_bv_corners], axis=0)
        coll_mat = box_collision_test(coll_query_matrix, coll_query_matrix)
        n_gt, tmp_bboxes = len(avoid_coll_boxes_bv_corners), []
        for i in range(n_gt, len(coll_mat)):
            if any(coll_mat[i]):
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                cur_sample = sampled_cls_list[i - n_gt]
                pt_path = os.path.join(data_root, cur_sample['path'])
                sampled_pts_cur = read_points(pt_path)
                sampled_pts_cur[:, :3] += cur_sample['box3d_lidar'][:3]
                sampled_pts.append(sampled_pts_cur)
                sampled_names.append(cur_sample['name'])
                sampled_labels.append(CLASSES[cur_sample['name']])
                sampled_bboxes.append(cur_sample['box3d_lidar'])
                tmp_bboxes.append(cur_sample['box3d_lidar'])
                sampled_difficulty.append(cur_sample['difficulty'])
        if len(tmp_bboxes) == 0:
            tmp_bboxes = np.array(tmp_bboxes).reshape(-1, 7)
        else:
            tmp_bboxes = np.array(tmp_bboxes)
        avoid_coll_boxes = np.concatenate([avoid_coll_boxes, tmp_bboxes], axis=0)
        
    # merge sampled database
    # remove raw points in sampled_bboxes firstly
    pts = remove_pts_in_bboxes(pts, np.stack(sampled_bboxes, axis=0))
    # pts = np.concatenate([pts, np.concatenate(sampled_pts, axis=0)], axis=0)
    pts = np.concatenate([np.concatenate(sampled_pts, axis=0), pts], axis=0)
    gt_bboxes_3d = avoid_coll_boxes.astype(np.float32)
    gt_labels = np.concatenate([gt_labels, np.array(sampled_labels)], axis=0)
    gt_names = np.concatenate([gt_names, np.array(sampled_names)], axis=0)
    difficulty = np.concatenate([gt_difficulty, np.array(sampled_difficulty)], axis=0)
    data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels, 
            'gt_names': gt_names,
            'difficulty': difficulty,
            'image_info': image_info,
            'calib_info': calib_info
        }
    return data_dict


@numba.jit(nopython=True)
def object_noise_core(pts, gt_bboxes_3d, bev_corners, trans_vec, rot_angle, rot_mat, masks):
    '''
    pts: (N, 4)
    gt_bboxes_3d: (n_bbox, 7)
    bev_corners: ((n_bbox, 4, 2))
    trans_vec: (n_bbox, num_try, 3)
    rot_mat: (n_bbox, num_try, 2, 2)
    masks: (N, n_bbox), bool
    return: gt_bboxes_3d, pts
    '''
    # 1. select the noise of num_try for each bbox under the collision test
    n_bbox, num_try = trans_vec.shape[:2]
    
    # succ_mask: (n_bbox, ), whether each bbox can be added noise successfully. -1 denotes failure.
    succ_mask = -np.ones((n_bbox, ), dtype=np.int_)
    for i in range(n_bbox):
        for j in range(num_try):
            cur_bbox = bev_corners[i] - np.expand_dims(gt_bboxes_3d[i, :2], 0) # (4, 2) - (1, 2) -> (4, 2)
            rot = np.zeros((2, 2), dtype=np.float32)
            rot[:] = rot_mat[i, j] # (2, 2)
            trans = trans_vec[i, j] # (3, )
            cur_bbox = cur_bbox @ rot
            cur_bbox += gt_bboxes_3d[i, :2]
            cur_bbox += np.expand_dims(trans[:2], 0) # (4, 2)
            coll_mat = box_collision_test(np.expand_dims(cur_bbox, 0), bev_corners)
            coll_mat[0, i] = False
            if coll_mat.any():
                continue
            else:
                bev_corners[i] = cur_bbox # update the bev_corners when adding noise succseefully.
                succ_mask[i] = j
                break
    # 2. points and bboxes noise
    visit = {}
    for i in range(n_bbox):
        jj = succ_mask[i] 
        if jj == -1:
            continue
        cur_trans, cur_angle = trans_vec[i, jj], rot_angle[i, jj]
        cur_rot_mat = np.zeros((2, 2), dtype=np.float32)
        cur_rot_mat[:] = rot_mat[i, jj]
        for k in range(len(pts)):
            if masks[k][i] and k not in visit:
                cur_pt = pts[k] # (4, )
                cur_pt_xyz = np.zeros((1, 3), dtype=np.float32)
                cur_pt_xyz[0] = cur_pt[:3] - gt_bboxes_3d[i][:3]
                tmp_cur_pt_xy = np.zeros((1, 2), dtype=np.float32)
                tmp_cur_pt_xy[:] = cur_pt_xyz[:, :2]
                cur_pt_xyz[:, :2] = tmp_cur_pt_xy @ cur_rot_mat # (1, 2)
                cur_pt_xyz[0] = cur_pt_xyz[0] + gt_bboxes_3d[i][:3]
                cur_pt_xyz[0] = cur_pt_xyz[0] + cur_trans[:3]
                cur_pt[:3] = cur_pt_xyz[0]
                visit[k] = 1

        gt_bboxes_3d[i, :3] += cur_trans[:3]
        gt_bboxes_3d[i, 6] += cur_angle

    return gt_bboxes_3d, pts


def object_noise(data_dict, num_try, translation_std, rot_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    num_try: int, 100
    translation_std: shape=[3, ]
    rot_range: shape=[2, ]
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    n_bbox = len(gt_bboxes_3d)
    
    # 1. generate rotation vectors and rotation matrices
    trans_vec = np.random.normal(scale=translation_std, size=(n_bbox, num_try, 3)).astype(np.float32)
    rot_angle = np.random.uniform(rot_range[0], rot_range[1], size=(n_bbox, num_try)).astype(np.float32)
    rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
    # in fact, - rot_angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2, n_bbox, num_try)
    rot_mat = np.transpose(rot_mat, (2, 3, 1, 0)) # (n_bbox, num_try, 2, 2)
    
    # 2. generate noise for each bbox and the points inside the bbox.
    bev_corners = bbox3d2bevcorners(gt_bboxes_3d) # (n_bbox, 4, 2) # for collision test
    masks = remove_pts_in_bboxes(pts, gt_bboxes_3d, rm=False) # identify which point should be added noise
    gt_bboxes_3d, pts = object_noise_core(pts=pts, 
                                          gt_bboxes_3d=gt_bboxes_3d, 
                                          bev_corners=bev_corners, 
                                          trans_vec=trans_vec, 
                                          rot_angle=rot_angle, 
                                          rot_mat=rot_mat, 
                                          masks=masks)
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'pts': pts})

    return data_dict


def random_flip(data_dict, random_flip_ratio):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    random_flip_ratio: float, 0-1
    return: data_dict
    '''
    random_flip_state = np.random.choice([True, False], p=[random_flip_ratio, 1-random_flip_ratio])
    if random_flip_state:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        pts[:, 1] = -pts[:, 1] 
        gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
        gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] + np.pi
        data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
        data_dict.update({'pts': pts})
    return data_dict


def global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    rot_range: [a, b]
    scale_ratio_range: [c, d] 
    translation_std:  [e, f, g]
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    
    # 1. rotation
    rot_angle = np.random.uniform(rot_range[0], rot_range[1])
    rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
    # in fact, - rot_angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2)
    # 1.1 bbox rotation
    gt_bboxes_3d[:, :2] = gt_bboxes_3d[:, :2] @ rot_mat.T
    gt_bboxes_3d[:, 6] += rot_angle
    # 1.2 point rotation
    pts[:, :2] = pts[:, :2] @ rot_mat.T

    # 2. scaling
    scale_fator = np.random.uniform(scale_ratio_range[0], scale_ratio_range[1])
    gt_bboxes_3d[:, :6] *= scale_fator
    pts[:, :3] *= scale_fator

    # 3. translation
    trans_factor = np.random.normal(scale=translation_std, size=(1, 3))
    gt_bboxes_3d[:, :3] += trans_factor
    pts[:, :3] += trans_factor
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'pts': pts})
    return data_dict


def point_range_filter(data_dict, point_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    pts = data_dict['pts']
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    data_dict.update({'pts': pts})
    return data_dict 


def object_range_filter(data_dict, object_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], data_dict['gt_labels']
    gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']

    # bev filter
    flag_x_low = gt_bboxes_3d[:, 0] > object_range[0]
    flag_y_low = gt_bboxes_3d[:, 1] > object_range[1]
    flag_x_high = gt_bboxes_3d[:, 0] < object_range[3]
    flag_y_high = gt_bboxes_3d[:, 1] < object_range[4]
    keep_mask = flag_x_low & flag_y_low & flag_x_high & flag_y_high

    gt_bboxes_3d, gt_labels = gt_bboxes_3d[keep_mask], gt_labels[keep_mask]
    gt_names, difficulty = gt_names[keep_mask], difficulty[keep_mask]
    gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], 0.5, 2 * np.pi)
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'gt_labels': gt_labels})
    data_dict.update({'gt_names': gt_names})
    data_dict.update({'difficulty': difficulty})
    return data_dict


def points_shuffle(data_dict):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    '''
    pts = data_dict['pts']
    indices = np.arange(0, len(pts))
    np.random.shuffle(indices)
    pts = pts[indices]
    data_dict.update({'pts': pts})
    return data_dict


def filter_bboxes_with_labels(data_dict, label=-1):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    label: int
    '''
    gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], data_dict['gt_labels']
    gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']
    idx = gt_labels != label
    gt_bboxes_3d = gt_bboxes_3d[idx]
    gt_labels = gt_labels[idx]
    gt_names = gt_names[idx]
    difficulty = difficulty[idx]
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'gt_labels': gt_labels})
    data_dict.update({'gt_names': gt_names})
    data_dict.update({'difficulty': difficulty})
    return data_dict

def data_augment(CLASSES, data_root, data_dict, data_aug_config):
    '''
    CLASSES: dict(Pedestrian=0, Cyclist=1, Car=2)
    data_root: str, data root
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    data_aug_config: dict()
    return: data_dict
    '''


    # 1. sample databases and merge into the data 
    db_sampler_config = data_aug_config['db_sampler']
    data_dict = dbsample(CLASSES,
                         data_root,
                         data_dict, 
                         db_sampler=db_sampler_config['db_sampler'],
                         sample_groups=db_sampler_config['sample_groups'])
    
    # 2. apply octodron dropout
    data_dict = octodron_dropout(data_dict)


    # 3. object noise
    object_noise_config = data_aug_config['object_noise']
    data_dict = object_noise(data_dict, 
                             num_try=object_noise_config['num_try'],
                             translation_std=object_noise_config['translation_std'],
                             rot_range=object_noise_config['rot_range'])
    
    # 4. random flip
    random_flip_ratio = data_aug_config['random_flip_ratio']
    data_dict = random_flip(data_dict, random_flip_ratio)

    # 5. global rotation, scaling and translation
    global_rot_scale_trans_config = data_aug_config['global_rot_scale_trans']
    rot_range = global_rot_scale_trans_config['rot_range']
    scale_ratio_range = global_rot_scale_trans_config['scale_ratio_range']
    translation_std = global_rot_scale_trans_config['translation_std']
    data_dict = global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std)

    # 6. points range filter
    point_range = data_aug_config['point_range_filter']
    data_dict = point_range_filter(data_dict, point_range)

    # # 7. object range filter
    object_range = data_aug_config['object_range_filter']
    data_dict = object_range_filter(data_dict, object_range)

    # 8. points shuffle
    data_dict = points_shuffle(data_dict)

    # # 9. filter bboxes with label=-1
    # data_dict = filter_bboxes_with_labels(data_dict)
    
    return data_dict
