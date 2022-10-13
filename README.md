# action-chore

## Start work
```
# ! /usr/bin/bash
srun --pty /bin/bash
setw -g mode-mouse on

module load cuda/11.3
module load anaconda3/2022.01

# for chore env, use gcc/9.2.0
module load gcc/9.2.0

source activate chore

sinfo -p jiang --Format=time,nodes,statecompact,gres
squeue -p jiang
scancel JOB_ID
```

## chore code

```

# test pytorch3d
python -W ignore pytorch3d_test.py

# download pretrained model
wget -O chore-pretrained.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/SatwEeqFnQdBGaF/download
unzip chore-pretrained.zip -d experiments

# run demp
python -W ignore demo.py chore-release -s example -on basketball 

# preprocess
python -W ignore preprocess/preprocess_scale.py -s /work/vig/qixinhu/repo/BEHAVE-dataset/sequences/Date01_Sub01_backpack_back
python -W ignore preprocess/preprocess_scale.py -a

# train
python -W ignore -m torch.distributed.launch --nproc_per_node=4 --use_env train_launch.py -en chore-release

# testing

# Evaluate
unzip behave-test-object-fullmask.zip -d /work/vig/qixinhu/repo/BEHAVE-dataset/sequences

python recon/evaluate.py 

# first you need to generate reconstruction results
python -W ignore recon/recon_fit_behave.py chore-release --save_name chore-release -s /work/vig/qixinhu/repo/BEHAVE-dataset/sequences/Date03_Sub05_toolbox

python -W ignore recon/evaluate.py
```


## action-chore

```bash
# clone this repo
git clone https://github.com/qixinhu11/action-chore.git

# prepare action label
python preprocess/preprocess_action.py
```


```bash
# train action-chore-encoder
python -W ignore -m torch.distributed.launch --nproc_per_node=4 --use_env action_encoder_train_launch.py -en action-encoder-core

# train action-chore-decoder
python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=4 --use_env action_decoder_train_launch.py -en action-decoder-core
```


```bash
Testing
# first need to fit the the reconstruction
python -W ignore recon/recon_fit_behave_encoder.py action-encoder-core --save_name action-encoder-core -s /work/vig/qixinhu/repo/BEHAVE-dataset/sequences/Date03_Sub05_toolbox
python -W ignore recon/recon_fit_behave_decoder.py action-decoder-core --save_name action-decoder-core -s /work/vig/qixinhu/repo/BEHAVE-dataset/sequences/Date03_Sub05_toolbox

# evaluate on whole sequence
python -W ignore recon/recon_fit_behave_decoder.py action-decoder-core --save_name action-decoder-core
python -W ignore recon/recon_fit_behave_encoder.py action-encoder-core --save_name action-encoder-core
# and then evaluate its performance

```