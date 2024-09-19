import argparse
import os
import sys
import numpy as np

sys.path.append('/media/asghar/FutureDet/Projects/FutureDet')
sys.path.append('/media/asghar/FutureDet/Projects/Core/nuscenes-forecast/python-sdk')

import torch
from det3d.datasets.nuscenes.nuscenes import modify_forecast_boxes
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (batch_processor)
from det3d.torchie.trainer import load_checkpoint
import time
from det3d.core.input.voxel_generator import VoxelGenerator

ConfigP = '/media/asghar/FutureDet/Projects/FutureDet/configs/centerpoint/nusc_centerpoint_forecast_n3dtf_detection.py'
work_dirP = '/media/asghar/FutureDet/Projects/FutureDet/models/FutureDetection/nusc_centerpoint_forecast_n3dtf_detection'
checkpointP = '/media/asghar/FutureDet/Projects/FutureDet/models/FutureDetection/nusc_centerpoint_forecast_n3dtf_detection/latest.pth'
rootP = '/media/asghar/FutureDet/Coustom_Sample/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", default=ConfigP, help="train config file path")
    parser.add_argument("--work_dir", default=work_dirP, help="the dir to save logs and models")
    parser.add_argument("--checkpoint", default=checkpointP, help="the dir to checkpoint which the model read from")
    parser.add_argument("--root", default=rootP)
    parser.add_argument("--txt_result", type=bool, default=False,
                        help="whether to save results to standard KITTI format of txt type")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--extractBox", action="store_true", default=True)
    parser.add_argument("--forecast", type=int, default=7)
    parser.add_argument("--forecast_mode", default="velocity_dense")
    parser.add_argument("--classname", default="car")
    parser.add_argument("--rerank", default="last")
    parser.add_argument("--tp_pct", type=float, default=0.6)
    parser.add_argument("--static_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--cohort_analysis", default='extractBox', action="store_true")
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--association_oracle", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--nogroup", action="store_true")
    parser.add_argument("--K", default=1, type=int)
    parser.add_argument("--C", default=1, type=float)
    parser.add_argument("--split", default="mini_val")
    parser.add_argument("--version", default="v1.0-mini")  # v1.0-trainval   #v1.0-mini
    parser.add_argument("--modelCheckPoint", default="latest")
    parser.add_argument("--elapse_time", default=0.5)
    parser.add_argument("--num_input_features", default=5)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_' + path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1] + '.npy')
        points = np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]  # remove ring_index from features
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def load_cloud_from_nuscenes_file(pc_f,num_features):
    cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
    # last dimension should be the timestamp.
    cloud[:, 4] = 0
    return cloud


class Processor:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None

    def initialize(self):
        self.read_config()

    def read_config(self):
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

        #   self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        #  self.net = self.net.to(self.device).eval()
        load_checkpoint(self.net, self.model_path, map_location="cuda:0")
        self.net = self.net.cuda()
        self.net = self.net.eval()
        self.num_features = cfg.model.reader.num_input_features
        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[1],
        )

    def run(self, points):
        t_t = time.time()
        print(f"input points shape: {points.shape}")
        self.points = points.reshape([-1, self.num_features])
        self.points[:, 4] = 0  # timestamp value

        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)

        self.inputs = dict(
            voxels=voxels,
            num_points=num_points,
            num_voxels=num_voxels,
            coordinates=coords,
            shape=[grid_size],
            bev_map=[torch.from_numpy(np.zeros((1, 180, 180)))],
        )
        cpu_device = torch.device("cpu")
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = batch_processor(self.net, self.inputs, train_mode=False, local_rank=args.local_rank)
            outputs[0]['metadata'] = {'num_point_features': 5, 'timesteps': 7}
            for output in outputs:
                for k, v in output.items():
                    if k not in ["metadata", ]:
                        output[k] = v.to(cpu_device)
        det_boxes, tokens = modify_forecast_boxes(args.elapse_time, output, args.forecast, args.forecast_mode,
                                                  args.classname, args.jitter, args.K, args.C)
        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)
        return det_boxes


args  = parse_args()
cfg   = Config.fromfile(args.config)
clode = load_cloud_from_nuscenes_file(args.root, cfg.model.reader.num_input_features)
proc = Processor(args.config, args.checkpoint)
proc.initialize()
outputs = proc.run(clode)
print(outputs)
