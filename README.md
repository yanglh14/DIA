<h1> Dynamic Manipulation for Deformable Objecsts Leveraging Interaction</h1>

# Table of Contents
- 1 [Simulation](#simulation)
    - 1.1 [Setup](#setup)
    - 1.2 [Train VCD](#train-vcd)
    - 1.3 [Planning with VCD](#plan-vcd)
    - 1.4 [Graph Imitation Learning](#graph-imit)
    - 1.5 [Pretrained Models](#pretrained)
    - 1.6 [Demo](#Demo)
----
# Simulation

## Setup
This repository is a subset of [SoftAgent](https://github.com/Xingyu-Lin/softagent) cleaned up for VCD. Environment setup for VCD is similar to that of softagent.
1. Install [SoftGym](https://github.com/Xingyu-Lin/softgym). Then, copy softgym as a submodule in this directory by running `cp -r [path to softgym] ./`. Use the updated softgym on the vcd branch by `cd softgym && git checkout vcd`
2. You should have a conda environment named `softgym`. Install additional packages required by VCD, by `conda env update --file environment.yml` 
3. Generate initial environment configurations and cache them, by running `python VCD/generate_cached_initial_state.py`.
4. Run `./compile_1.0.sh && . ./prepare_1.0.sh` to compile PyFleX and prepare other paths.

## Train VCD
* Generate the dataset for training by running
    ```
    python VCD/main.py --gen_data=1 --dataf=./data/vcd
    ```
  Please refer to `main.py` for argument options.

* Train the dynamics model by running
    ```
    python VCD/main.py --gen_data=0 --dataf=./data/vcd_dyn
    ```
* Train the EdgeGNN model by running
    ```
    python VCD/main_train_edge.py --gen_data=0 --dataf=./data/vcd_edge
    ```
## Planning with VCD
```
python VCD/main_plan.py --edge_model_path={path_to_trained_edge_model}\
                        --partial_dyn_path={path_to_trained_dynamics_model}
```
An example for loading the model trained for 120 epochs:
```
python VCD/main_plan.py --edge_model_path ./data/vcd_edge/vsbl_edge_120.pth\
                        --partial_dyn_path ./data/vcd_dyn/vsbl_dyn_120.pth
```
## Graph Imitation Learning
1. Train dynamics using the full mesh
```
python VCD/main.py --gen_data=0 --dataf=./data/vcd --train_mode=full
```
2. Train dynamics using partial point cloud and imitate the teacher model
```
python VCD/main.py --gen_data=0 --dataf=./data/vcd --train_mode=graph_imit --full_dyn_path={path_to_teacher_model}
```

## Pretrained Model
Please refer to [this page](pretrained/README.md) for downloading the pretrained models.

If you find this codebase useful in your research, please consider citing:
