# iNeRF applied to Cube Tossing trajectory for CIS 6800 Final Project

Forked from https://github.com/yenchenlin/iNeRF-public, README for that project is in `original_iNeRF_README.md`, which
contains many important details about the repository.

Additionally, many important details on training the NeRF can be found at https://github.com/sxyu/pixel-nerf.

This README will only contain details for running the part of the project relevant to the CIS 6800 Final Project.

## Cube Tossing Dataset

Files from the cube tossing dataset can be downloaded from this google drive link: https://drive.google.com/drive/folders/12eLxVUQB0MxatJtPw7ExiI78WHCSxMiz?usp=share_link.
*Requires seas@upenn email.
The dataset has not yet been used in any publications and was generated from members of DAIRlab for future work on learning contact dynamics from tossing trajectories.

Extract the dataset into the `data/nerf_contactnets_data` folder.

The dataset contains rgb and depth images that have the background and robot arm already masked out.

Dataset is separate by the toss number (five total)
Masked RGB: `masked_rgb/`
Masked depth (unused): `masked_depth/`
Camera Extrinsics: `cam_K`

Dataloader: `src/data/ContactNetsDataset.py`

## Training NeRF

`python train/train.py -n <experiment_name> -c conf/exp/contactnets.conf -D data/nerf_contactnets_data --resume`

visualizations are generated in the `visuals/<experient_name>` folder

iNeRF experiment notebook

`pose_estimation.ipynb`

