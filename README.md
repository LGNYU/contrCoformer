# conformal-prediction

## Instructions
- Clone the repository and create a conda environment using the provided `conda_env.yml` file.
```
conda env create -f conda_env.yml
```
- Activate the environment using `conda activate baku`.
- Install the xarm environment using the following command.
```
cd xarm_env
pip install -e .
```
- For optimal transport based distance metrics, install the POT library using the following command.
```
pip install POT==0.7.0
```

- Place demonstration pkl files in `path/to/repo/expert_demos/xarm_env/`. Set `root_dir` in `cfg/config_eval.yaml` to `path/to/repo`.

- `cd baku` and use the following command to run the code for comparing 2 trajectories
```
python compare_traj.py suite=xarm_env dataloader=xarm_env experiment=compare_traj eval=false num_queries=20 bc_weight=<weight_path>
```