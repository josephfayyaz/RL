# 🦿 Reinforcement Learning for Custom Hopper with Domain Randomization

This repository presents a comprehensive study on **reinforcement learning (RL)** algorithms applied to a **custom MuJoCo Hopper** environment. Our goal is to develop robust control policies through **domain randomization**, **curriculum learning**, and **policy gradient methods**.

The project combines classic and modern RL techniques:
- **REINFORCE & Actor-Critic** (from-scratch implementations)
- **Proximal Policy Optimization (PPO)** (via Stable-Baselines3)
- **Uniform Domain Randomization (UDR)** and **Entropy-based Curriculum Domain Randomization (ES-CDR)**

---

## 📁 Repository Structure

```bash
.
├── agents/                              # (Optional) future agents folder
├── env/                                 # Custom MuJoCo environment
│   ├── assets/
│   ├── __init__.py
│   ├── custom_hopper.py                 # Custom Hopper environment with DR
│   └── mujoco_env.py
│
├── Logs/                                # CSV logs and WandB (if used)
│   ├── actor_critic/
│   ├── baseline/
│   ├── PPO_episode_rewards/
│   └── PPO_runtime_tmp/
│
├── models/                              # Trained model checkpoints
│   ├── actor_critic/
│   ├── PPO_runtime_tmp/
│   └── reinforce_baseline/
│
├── evaluation/                              # Optional rendering or video files
│
├── training/                            # Main training scripts
│   ├── wandb/                           # WandB config/data (if used)
│   ├── PPO_Hyperparameter_Calculation.py
│   ├── Train_PPO_UDR_ES_CDR.py
│   ├── Train_Actor_Critic.py
│   ├── Train_Reinforce_Baseline.py
│   └── Train_Reinforce_vanila.py
│
├── utils/
│   └── metric_extraction.py             # CSV parsing, plotting utilities
│
├── wandb/                               # WandB cache directory (gitignored)
├── .gitignore
├── __init__.py
└── README.md
```

---

## 🧪 Environments & Randomization

The environment is based on a custom subclass of the MuJoCo Hopper (`custom_hopper.py`), extended with:

- **Parameter Randomization**: friction, damping, body mass, initial state
- **Domain Randomization**:
  - *Uniform DR (UDR)*: randomized every episode
  - *Curriculum DR (ES-CDR)*: difficulty scaled with agent performance and return entropy

---

## 🧠 Algorithms Implemented

| Algorithm              | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| **REINFORCE**           | Monte Carlo policy gradient with optional baseline                 |
| **Actor-Critic**        | TD-based policy/value method                                       |
| **PPO**                 | Clipped surrogate objective with GAE (Stable-Baselines3)           |
| **UDR**                 | Domain variation with uniform sampling                             |
| **ES-CDR**              | Return entropy-driven difficulty adjustment                        |

---

## 🛠️ Setup Instructions

### ✅ Dependencies (tested on Ubuntu 22.04)

```bash
pip install -r requirements.txt
```

Essential packages:

```text
gym==0.21.0
stable-baselines3==1.7.0
mujoco-py>=2.1,<2.2
cython<3
scipy
Jinja2==3.1.2
importlib-metadata==4.13.0
patchelf
```

Make sure MuJoCo is installed and licensed correctly.

---

## 🚀 Training Commands

### 🎯 REINFORCE / Actor-Critic

```bash
python training/Train_Reinforce_vanila.py
python training/Train_Actor_Critic.py
python training/Train_Reinforce_Baseline.py
```

### 🤖 PPO with UDR + ES-CDR

```bash
python training/Train_PPO_UDR_ES_CDR.py --seed 0 --train_steps 350000
```

### 🔬 PPO Hyperparameter Sweep

```bash
python training/PPO_Hyperparameter_Calculation.py
```

---

## 📊 Logging & Evaluation

- All CSV logs are stored in `Logs/csv/`
- Trained models saved under `Models/`
- You can visualize performance using:

```python
from evaluation.plot_csv_scripts.metric_extraction import plot_training_curve

plot_training_curve("Logs/PPO_episode_rewards/ppo_run.PPO_episode_rewards")
```

---

## 📈 Sample Results

| Curriculum Level | Avg Return | Std Dev | Entropy |
|------------------|------------|---------|---------|
| Level 1          | 810        | 60      | 0.95    |
| Level 2          | 720        | 82      | 1.20    |
| Level 3          | 680        | 90      | 1.50    |

---

## 👥 Project Participants

Please list contributors here:

- **Pishool** – Custom environment design, algorithm implementation, evaluation
- *(Add names and roles as needed)*

---

## 📚 References & Acknowledgements

- OpenAI Baselines
- Stable-Baselines3 Docs
- MuJoCo Documentation

---

## 🧠 Future Directions

- Add evaluation over unseen dynamics
- Experiment with off-policy algorithms (e.g., SAC, DDPG)
- Integrate video rendering and performance visualizations

---

## 📬 Contact

Please reach out via GitHub issues or Linkedin profiles. 
- **Ali Vaezi** - [LinkedIn](https://www.linkedin.com/in/aliivaezii/)
- **Yousef Fayyaz** - [LinkedIn](https://www.linkedin.com/in/yousef-fayyaz-55ab9a255/)
- **Sajjad Shahali** - [LinkedIn](https://www.linkedin.com/in/sajjad-shahali/)
- **Parastoo Hashemi Alvar** - [LinkedIn](https://www.linkedin.com/in/parastoo-hashemi/)
