import copy
from pathlib import Path
import pickle

import fire, os
import sys

sys.path.append('/media/asghar/media/DAT')
sys.path.append('/media/asghar/media/DAT/Core/nuscenes-forecast/python-sdk')

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds
#root_path = "/media/asghar/media/NUSCENES_DATASET_ROOT_MINI"
root_path = "/media/asghar/media/NUSCENES_DATASET_ROOT"

#version = 'v1.0-mini'
version = 'v1.0-trainval'

def nuscenes_data_prep(root_path, version, experiment="trainval_forecast", nsweeps=20, filter_zero=True, timesteps=7):
    past = True if "past" in experiment else False

    if not os.path.isdir(root_path + "/" + experiment):
        os.makedirs(root_path + "/" + experiment)

    nu_ds.create_nuscenes_infos(root_path, version=version, experiment=experiment, nsweeps=nsweeps,
                                filter_zero=filter_zero, timesteps=timesteps, past=past)
    create_groundtruth_database(
        "NUSC",
        root_path + "/{}".format(experiment),
        Path(root_path) / "{}/infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(experiment, nsweeps, filter_zero),
        nsweeps=nsweeps,
        timesteps=timesteps
    )


def waymo_data_prep(root_path, split, nsweeps=1):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train':
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )


if __name__ == "__main__":
    fire.Fire()


nuscenes_data_prep(root_path,version)