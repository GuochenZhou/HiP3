# Hierarchical Pareto Policy Pool (HiP3)
This is the source code of Hierarchical Pareto Policy Pool. 

## Key Dependencies
```console
python==3.7.1
- git+https://github.com/Farama-Foundation/d4rl
- gym==0.23.1
- torch==1.8.0+cu111
- tensorflow==2.5.0
- mujoco==2.1.0
- mujoco-py==2.1.2.14
- numpy==1.21.0
```
## Usage
* Offline training
```console
python train_offline.py
```

* Online training (Please update the directory of the pre-trained policies to the papramter setting in train_online.py) 
```console
python train_online.py
```
