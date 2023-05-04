# Improvements on Data Augmentation, Backbone Network, and Loss Functions in PointPillars

PointPillars is an algorithm to make object detection with LiDAR data and it was a state of art model in 2019. We studied the original PointPillars paper and explored new alternatives to improved model performance based on other papers. We compared data augmentation methods including cross-scene points swapping and mixing as well as intra-scene dropout and mixing, added additional layers in the backbone layer, adjusted kernel sizes in the neck layer, and tested different loss functions. Our best model reached a better performance in the 2D BBOX, AOS, Bird's Eye View BBOX tasks than the original model at https://github.com/zhulf0804/PointPillars, while having similar level of performance on the 3D BBox task for easy and medium objects, and marginally worse performance for hard objects.

## [Env_Setup] 

```
Refer to the step by step environment setup guide in the /env_setup directory
```

## [Compile] 

```
cd ops
python setup.py develop
```

## [Datasets]

1. Download

    Download [point cloud](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)(29GB), [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)(12 GB), [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)(16 MB)和[labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)(5 MB)。Format the datasets as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7418 .bin)
    ```

2. Pre-process KITTI datasets First

    ```
    cd PointPillars/
    python pre_process_kitti.py --data_root your_path_to_kitti
    ```

    Now, we have datasets as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
            |- velodyne_reduced (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7518 .bin)
            |- velodyne_reduced (#7518 .bin)
        |- kitti_gt_database (# 19700 .bin)
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
    
    ```

## [Training]

```
cd PointPillars/
python train.py --data_root your_path_to_kitti
```

## [Evaluation]

```
cd PointPillars/
python evaluate.py --ckpt pretrained/epoch_160.pth --data_root your_path_to_kitti 
```

## [Test]

```
cd PointPillars/

# 1. infer and visualize point cloud detection
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path 

# 2. infer and visualize point cloud detection and gound truth.
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path  --gt_path your_gt_path

# 3. infer and visualize point cloud & image detection
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path --img_path your_img_path


e.g. [infer on val set 000134]

python test.py --ckpt pretrained/epoch_160.pth --pc_path /home/lifa/data/KITTI/training/velodyne_reduced/000134.bin

or

python test.py --ckpt pretrained/epoch_160.pth --pc_path /home/lifa/data/KITTI/training/velodyne_reduced/000134.bin --calib_path /home/lifa/data/KITTI/training/calib/000134.txt --img_path /home/lifa/data/KITTI/training/image_2/000134.png --gt_path /home/lifa/data/KITTI/training/label_2/000134.txt

```

## Acknowledements

Thanks for the open souce code [pointpillars](https://github.com/zhulf0804/PointPillars) [pcdet](https://github.com/open-mmlab/OpenPCDet) [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d).
