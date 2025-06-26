# 🦿 Reinforcement Learning for Custom Hopper with Domain Randomization

his repository presents a comprehensive study on **reinforcement learning (RL)** algorithms applied to a **custom MuJoCo Hopper** environment. The project aims to build robust locomotion policies under uncertain dynamics using:

- Classic Policy Gradient Methods: REINFORCE, Actor-Critic
- Advanced On-Policy Algorithms: Proximal Policy Optimization (PPO)
- Robustness Techniques: Domain Randomization (UDR), Curriculum Learning (CDR), Entropy Scheduling (ES)


## 🎥 Hopper Locomotion Policy Demo

<div align="center">
  <a href="https://youtu.be/TYBzPKied9g" target="_blank">
    <img src="render/plots/hopper_animation.gif" 
         alt="Watch the Hopper PPO + CDR + ES demo on YouTube" width="400"/>
  </a>
  <br/>
  <strong>Click to watch the full 15-second demo</strong>
</div>

---

## 📁 Repository Structure

```bash
.
├── src/                              # Core code (Python package)
│   ├── agents/                       # RL algorithm implementations
│   ├── env/                          # Custom MuJoCo-Hopper wrappers
│   ├── evaluation/                   # Metrics, plotting, helper scripts
│   └── training/                     # Training entry-points & configs
│
├── Logs/                             # Raw tensorboard/CSV logs
│   ├── Learning_Curve/               #  ⇢ learning-curve CSVs
│   ├── PPO_episode_rewards/          #  ⇢ per-episode returns
│   ├── PPO_robustness/               #  ⇢ domain-randomisation runs
│   ├── PPO_runtime_tmp/              #  ⇢ scratch & tmp logs
│   ├── actor_critic/                 #  ⇢ AC experiments
│   └── baseline/                     #  ⇢ REINFORCE baseline runs
│
├── models/                          
│   ├── PPO/
│   ├── actor_critic/
│   └── reinforce_baseline/
│
├── render/                           # Visual outputs (GIF/MP4/PNG)
│   └── plots/
│
├── requirements.txt                  # Python dependencies
├── README.md                         # You are here 👋
├── __init__.py                       # Makes repo import-able (`import rl_master`)
├── .idea/                            # IDE settings  (⇢ add to .gitignore)
└── __pycache__/                      # Byte-code cache (auto-generated)


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


## ⚙️ Environment Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

You’ll need MuJoCo 2.1+ properly installed and licensed. Refer to:
👉 https://github.com/openai/mujoco-py#install-mujoco

---

## 🧪 Training

From the root directory, run:

```bash
# REINFORCE
python src/training/Train_Reinforce_vanila.py

# REINFORCE with baseline
python src/training/Train_Baseline.py

# Actor-Critic
python src/training/Train_Actor_Critic.py

# PPO + UDR + ES-CDR
python src/training/PPO_UDR_ES_CDR.py --Domain cdr --Entropy_Scheduling True --seed 0
```

---

## 🔬 Hyperparameter Optimization

```bash
python src/training/PPO_Hyperparameter_Calculation.py
```

You can adjust sweep parameters via JSON or inline config.

---

## 📊 Logging & Visualization

Training metrics (returns, entropy, etc.) are saved as CSV in the `Logs/` directory.

To plot results:

```bash
python evaluation/plot_csv_scripts/plot_metrics.py
```

Or use the built-in metric utilities in `src/evaluation`.

---

## 🤖 Custom Environment

Implemented in `src/env/custom_hopper.py`, our environment introduces:

- Dynamic randomization of:
  - Mass, friction, damping, init pose
- UDR: Resampled every episode
- CDR + ES: Difficulty increases based on policy performance and entropy

---

## 🧠 PPO + Curriculum Domain Randomization (CDR) + Entropy Scheduling (ES)

This project extends PPO with **adaptive training difficulty** using:

### 🔁 Curriculum Domain Randomization (CDR)
CDR gradually increases the range of domain parameters (e.g., torso mass, friction) during training, helping the agent:
- First master simple dynamics.
- Then adapt to complex, realistic scenarios.

Use it with:
```bash
--Domain cdr
```

### 📉 Entropy Scheduling (ES)
ES monitors the **policy’s return entropy**. When the agent is confident (low entropy), it:
- Advances the curriculum level.
- Makes the environment harder.

Enable it with:
```bash
--Entropy_Scheduling True
```

### 🧪 Example PPO + CDR + ES Command
```bash
python src/training/PPO_UDR_ES_CDR.py --Domain cdr --Entropy_Scheduling True --seed 0
```

---

## 📈 Sample Results

| Level | Mean Return | Std Dev | Return Entropy |
|-------|-------------|---------|----------------|
| 1     | 820         | ±50     | 1.02           |
| 2     | 710         | ±70     | 1.30           |
| 3     | 665         | ±85     | 1.48           |

---
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
