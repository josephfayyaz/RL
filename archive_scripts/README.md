# Project on Reinforcement Learning (Course project MLDL 2025 - POLITO)
### Teaching assistants: Andrea Protopapa and Davide Buoso

Starting code for "Project 4: Reinforcement Learning" course project of MLDL 2025 at Polytechnic of Turin. Official assignment at [Google Doc](https://docs.google.com/document/d/16Fy0gUj-HKxweQaJf97b_lTeqM_9axJa4_SdqpP_FaE/edit?usp=sharing).


## Getting started

Before starting to implement your own code, make sure to:
1. read and study the material provided (see Section 1 in the assignment)
2. read the documentation of the main packages you will be using ([mujoco-py](https://github.com/openai/mujoco-py), [Gym](https://github.com/openai/gym), [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))
3. play around with the code in the template to familiarize with all the tools. Especially with the `test_random_policy.py` script.


### 1. Local on Linux (recommended)

if you have a Linux system, you can work on the course project directly on your local machine. By doing so, you will also be able to render the Mujoco Hopper environment and visualize what is happening. This code has been tested on Linux with python 3.7.

**Installation**
- (recommended) create a new conda environment, e.g. `conda create --name mldl pip=22 python=3.8 setuptools=65.5.0 wheel=0.38`
- Run `pip install -r requirements.txt`
- Install MuJoCo 2.1 and the Python Mujoco interface:
	- follow instructions here: https://github.com/openai/mujoco-py
	- see Troubleshooting section below for solving common installation issues.

Check your installation by launching `python test_random_policy.py`.


### 2. Local on Windows
As the latest version of `mujoco-py` is not compatible for Windows explicitly, you may:
- Try installing WSL2 (requires fewer resources) or a full Virtual Machine to run Linux on Windows. Then you can follow the instructions above for Linux.
- (not recommended) Try downloading a [previous version](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/) of `mujoco-py`.
- (not recommended) Stick to the Google Colab template (see below), which runs on the browser regardless of the operating system. This option, however, will not allow you to render the environment in an interactive window for debugging purposes.


### 3. Remotely on Google Colab

Alternatively, you may also complete the project on [Google Colab](https://colab.research.google.com/):

- Download the files contained in the `colab_template` folder in this repo.
- Load the `.ipynb` files on [https://colab.research.google.com/](colab) and follow the instructions on each script to run the experiments.

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once (e.g. using a Linux system), to best understand what is going on with the underlying Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.



## Troubleshooting
- General installation guide and troubleshooting: [Here](https://docs.google.com/document/d/1j5_FzsOpGflBYgNwW9ez5dh3BGcLUj4a/edit?usp=sharing&ouid=118210130204683507526&rtpof=true&sd=true)
- If having trouble while installing mujoco-py, see [#627](https://github.com/openai/mujoco-py/issues/627) to install all dependencies through conda.
- If installation goes wrong due to gym==0.21 as `error in gym setup command: 'extras_require'`, see https://github.com/openai/gym/issues/3176. There is a problem with the version of setuptools.
- if you get a `cannot find -lGL` error when importing mujoco_py for the first time, then have a look at my solution in [#763](https://github.com/openai/mujoco-py/issues/763#issuecomment-1519090452)
- if you get a `fatal error: GL/osmesa.h: No such file or directory` error, make sure you export the CPATH variable as mentioned in mujoco-py[#627](https://github.com/openai/mujoco-py/issues/627)
- if you get a `Cannot assign type 'void (const char *) except * nogil' to 'void`, then run `pip install "cython<3"` (see issue [#773](https://github.com/openai/mujoco-py/issues/773))