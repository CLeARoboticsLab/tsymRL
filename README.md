# Implementation of Time Symmetric Data Augmentation (TSDA) in PyTorch

This is PyTorch implementation of TSDA from

**An Investigation of Time Reversal Symmetry in Reinforcement Learning** by 

[Brett Barkley](https://bebark.github.io/), [Amy Zhang](https://amyzhang.github.io/), and [David Fridovich-Keil](https://clearoboticslab.github.io/).

Add paper link
<!-- [[Paper]](https://arxiv.org/abs/1910.01741) -->

This repository is built as an extension of the Pytorch implementation of 
**Improving Sample Efficiency in Model-Free Reinforcement Learning from Images** by

[Denis Yarats](https://cs.nyu.edu/~dy1042/), [Amy Zhang](https://mila.quebec/en/person/amy-zhang/), [Ilya Kostrikov](https://github.com/ikostrikov), [Brandon Amos](http://bamos.github.io/), [Joelle Pineau](https://www.cs.mcgill.ca/~jpineau/), [Rob Fergus](https://cs.nyu.edu/~fergus/pmwiki/pmwiki.php).

[[Paper]](https://arxiv.org/abs/1910.01741) [[Webpage]](https://sites.google.com/view/sac-ae/home)

## Citation
If you use this repo in your research, please consider citing the paper as follows
<!-- ```
@article{barkley2023TSDA,
    title={An Investigation of Time Reversal Symmetry in Reinforcement Learning},
    author={Brett Barkley and Amy Zhang and David Fridovich-Keil},
    year={2023},
    eprint={1910.01741},
    archivePrefix={arXiv}
}
``` -->

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running the following in the top level directory:
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with:
```
source activate tsda
```

## Instructions
To train an SAC+AE agent on the `cheetah run` task from image-based observations  run:
```
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir ./log \
    --seed 1
```
This will produce 'log' folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir log
```
and opening up tensorboad in your browser.

The console output is also available in a form:
```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000
```
a training entry decodes as:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
RLOSS - average reconstruction loss (only if is trained from pixels and decoder)
```
while an evaluation entry:
```
| eval | S: 0 | ER: 21.1676
```
which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 10).

## Results

Empirical evaluations showcase how the synthetic transitions provided by TSDA enhance the sample efficiency of RL agents in time reversible scenarios without friction or contact. In environments where the assumptions of TSDA are not globally satisfied we find that TSDA can significantly degrade
sample efficiency and policy performance, but can also improve sample efficiency under the
right conditions. Ultimately we conclude that time symmetry shows promise in enhancing
the sample efficiency of reinforcement learning if the environment
and reward structures are of an appropriate form for TSDA to be employed effectively.

One hypothesis of particular note that we mention in our paper is that TSDA can lead agents to favor early exploitation over exploration in time
symmetric environments, e.g., when important exploratory actions lie at the edge of the
action space. The two videos below depict the typical training progress in evaluation episodes with and without TSDA. 

<p align="center">
  <img width="49.5%" src="https://imgur.com/x0zsrTg.gif">
  <img width="49.5%" src="https://imgur.com/ouLYLJY.gif">
 </p>
<!-- Our method demonstrates significantly improved performance over the baseline SAC:pixel. It matches the state-of-the-art performance of model-based algorithms, such as PlaNet (Hafner et al., 2018) and SLAC (Lee et al., 2019), as well
as a model-free algorithm D4PG (Barth-Maron et al., 2018), that also learns from raw images. Our
algorithm exhibits stable learning across ten random seeds and is extremely easy to implement.
![Results](results/graph.png) -->
