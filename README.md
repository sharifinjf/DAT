# DAT
<video width="320" height="240" controls loop muted autoplay>
  <source src="https://raw.githubusercontent.com/sharifinjf/DAT/main/docs/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
![A](docs/pipline.svg)

### Requirements
- OS: Ubuntu 20.04.06
- Python: 3.7.17 
- PyTorch: 1.11.0+cu113
- spconv-cu113: 2.2.6
- CUDA: 11.3
- CMake: 3.25.2 or higher
### Basic Installation 

```bash
# Basic python libraries
python3.7.17 -m venv DAT
source DAT/bin/activate
git clone git@github.com:sharifinjf/DAT.git
pip install -r requirements.txt
```

#### CUDA Extensions

```bash
# Set the CUDA/CuDNN path (change the path to your own CUDA location) 
export PATH=/usr/local/cuda-10.1/bin:$PATH
export CUDA_ROOT=/usr/local/cuda-10.1
export CUDA_HOME=/usr/local/cuda-10.1
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
```

## Use DAT
Be sure to change the paths in configs and syspath in the following files:
- train.py
- evaluate.py
- trajectory.py
- visualize.py
- Comparison_visulize
- det3d/datasets/nuscenes/nuscenes.py
- tools/create_data.py
- tools/dist_test.py

### Benchmark Evaluation and Training

#### Prepare Data for Training and Evaluation 

```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

Data creation should be under the GPU environment.

```
# nuScenes 
#python tools/create_data.py nuscenes_data_prep --root_path NUSCENES_DATASET_ROOT --version v1.0-trainval --timesteps 7

```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── NUSCENES_DATASET_ROOT
      ├── samples       <-- key frames
      ├── sweeps        <-- frames without annotation
      ├── maps          <-- unused
      |── v1.0-trainval <-- metadata and annotations
      |__ trainval_forecast
          |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
          |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
          |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
          |── gt_database_10sweeps_withvelo <-- GT database 
```


Use the following command to start a distributed training and evaluation. The models and logs will be saved to ```models/CONFIG_NAME```. Results will be save to ```results/CONFIG_NAME``` 

#### Constant Velocity Model
```bash
# Cars
python train.py --experiment DAT --model forecast_n0

python evaluate.py --experiment DAT --model forecast_n0 --forecast_mode velocity_constant  --cohort_analysis --extractBox

# Pedestrians
python train.py --experiment DAT --model pedestrian_forecast_n0

python evaluate.py --experiment DAT --model forecast_n0 --forecast_mode velocity_constant  --cohort_analysis --classname pedestrian --extractBox
```

#### FaF*
```bash
# Cars
python train.py --experiment DAT --model forecast_n3

python evaluate.py --experiment DAT --model forecast_n3 --forecast_mode velocity_forward  --cohort_analysis --extractBox

# Pedestrians
python train.py --experiment DAT --model pedestrian_forecast_n3

python evaluate.py --experiment DAT --model forecast_n3 --forecast_mode velocity_forward  --cohort_analysis --classname pedestrian --extractBox
```

#### DAT
```bash
# Cars
python train.py --experiment DAT --model forecast_n3dtf

python evaluate.py --experiment DAT --model forecast_n3dtf --forecast_mode velocity_dense  --cohort_analysis --extractBox

python evaluate.py --experiment DAT --model forecast_n3dtf --forecast_mode velocity_dense  --cohort_analysis --K 5 --eval_only

# Pedestrians
python train.py --experiment DAT --model pedestrian_forecast_n3dtf

python evaluate.py --experiment DAT --model forecast_n3dtf --forecast_mode velocity_dense  --cohort_analysis --classname pedestrian --extractBox

python evaluate.py --experiment DAT --model forecast_n3dtf --forecast_mode velocity_dense  --cohort_analysis --K 5 --classname pedestrian --eval_only

```
#### Evaluation Parameters
```
extractBox -> Uses modelCheckPoint to run inference on GPUs and save results to disk
tp_pct -> TP percentage thresholds for ADE@TP % and FDE@TP %. Setting tp_pct to -1 returns AVG ADE/FDE over all TP threholds.
static_only -> Rescores stationary objects to have higher confidence. Result from Table 1.
eval_only -> Uses cached results to run evaluation
forecast_mode -> Detection association method. [Constant Velocity -> velocity_constant, FaF* -> velocity_forward, DAT -> velocity_dense]
classname -> Select class to evaluate. car and pedestrian currently supported.
rerank -> Assignment of forecasting score. [last, first, average]
cohort_analysis -> Reports evaluation metrics per motion subclass static/linear/nonlinear.
K -> topK evaluation, only useful for FutureDet
```

### [Pre-trained Models](https://drive.google.com/drive/folders/18fvE3MvXQGDThIPs9iFjQMuUJp5_waxp?usp=sharing)

## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples below.
* [TrajectoryNAS]((https://github.com/sharifinjf/TrajectoryNAS))
* [FutuerDet](https://github.com/neeharperi/FutureDet/tree/main)
* [det3d](https://github.com/poodarchu/det3d)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [CenterNet](https://github.com/xingyizhou/CenterNet) 
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [CenterPoint](https://github.com/tianweiy/CenterPoint)


