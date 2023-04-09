# Project environment setup in Docker container

Steps to set up the development environment inside the container after the container is successfully spawned.
You need to first run 'docker attach  {HASH}' to go into the docker, before running the following. 

**** if you are not able to run the docker commands and see permissions denied error, it is because the docker was setup running as su, instead of the user. Run 'sudo su' to go into su mode will solve the permissions issue ****

# 1. Development environment setup

#### 1.1 Linux package download:
  - apt-get update. 
  - apt-get install -y build-essential git zip unzip software-properties-common python3-pip python3-dev gcc-multilib valgrind portmap rpcbind libcurl4-openssl-dev bzip2 libssl-dev llvm net-tools libtool pkg-config libgl1

#### 1.2 Python venv setup:
  - apt install python3.8-venv
  - python -m venv pillars --system-site-packages
  - switch to the 'pillars' venv by running the following command in the project root dir:
    - source pillars/bin/activate
  - install requirements
    - pip install -r requirements.txt (open3d needs to be 0.13.0, numpy needs to be numpy==1.20.3)
    - pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  - build pointpillars.egg==info: 
    - cd ops
    - python setup.py develop

#### 1.3 Prepare data
  - Step 1: mkdir data/kitti, cd to data
  - Step 2: run curl to download data (you may need to apt install curl): 
    - curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip -o data_object_velodyne.zip
    - curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip -o data_object_image_2.zip
    - curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip -o data_object_calib.zip 
    - curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip -o data_object_label_2.zip
  - Step 3: Unzip the data to data/kitti. Run commands in /data:
    - unzip -o data_object_velodyne.zip -d kitti
    - unzip -o data_object_image_2.zip -d kitti
    - unzip -o data_object_calib.zip -d kitti
    - unzip -o data_object_label_2.zip -d kitti
  - Step 4: preprocess the data:
    - cd to root dir of the project
    - python pre_process_kitti.py --data_root your_path_to_kitti

#### 1.4 train and eval
  - train: 
    - python train.py --data_root your_path_to_kitti
  - eval:
    - python evaluate.py --ckpt pretrained/epoch_160.pth --data_root your_path_to_kitti 

#### 1.5 test

  - 1. infer and visualize point cloud detection
    - python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path 

  - 2. infer and visualize point cloud detection and gound truth.
    - python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path  --gt_path your_gt_path

  - 3. infer and visualize point cloud & image detection
    - python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path --img_path your_img_path
    - e.g. [infer on val set 000134]
python test.py --ckpt pretrained/epoch_160.pth --pc_path /home/lifa/data/KITTI/training/velodyne_reduced/000134.bin
or
python test.py --ckpt pretrained/epoch_160.pth --pc_path /home/lifa/data/KITTI/training/velodyne_reduced/000134.bin --calib_path /home/lifa/data/KITTI/training/calib/000134.txt --img_path /home/lifa/data/KITTI/training/image_2/000134.png --gt_path /home/lifa/data/KITTI/training/label_2/000134.txt

