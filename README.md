# Reinforcement Learning for Liability-Driven Investment (LDI)

## About the Project

This project implements a sophisticated framework for applying **deep reinforcement learning (DRL)** to portfolio management in a **Liability-Driven Investment (LDI)** context. The core objective is to train an AI agent to make dynamic asset allocation decisions that ensure a pension fund's assets can meet its liabilities over time, thereby minimizing the risk of a funding shortfall.

## Key Innovations

- **LDI-Specific Simulation Environment**: The agent learns in an environment that combines economic scenario projections (the asset market) with fixed liability cash flow models.
- **Liability-Driven Reward Function**: The reward is designed to maximize the assets-to-liabilities ratio, heavily penalizing any scenario where the fund's value falls short of its obligations.
- **Asset Allocation Action Space**: The agent learns to rebalance a portfolio across asset classes like equities and fixed income to optimize for both return and liability matching.
- **Focus on Long-Term Solvency**: The model is trained to learn robust strategies that ensure the fund's solvency over multi-year simulated episodes.

---

## Project Structure
LDI_project/
├── data/
│   ├── raw/                     # Raw economic data and liability projections
│   └── processed/               # Processed features of the market and fund
├── src/
│   ├── data/                    # Data processing modules
│   │   └── processor.py         # Processor for economic and liability data
│   ├── environment/             # LDI environments
│   │   └── ldi_env.py           # The LDI simulation environment
│   ├── evaluation/              # Backtesting and evaluation
│   │   └── backtest.py          # Backtest framework for the LDI strategy
│   ├── training/                # Training algorithms
│   │   └── train.py             # PPO/SAC training for LDI
│   └── utils/                   # Utility functions
├── models/                      # Saved models
├── results/                     # Backtest results
│   ├── figures/                 # Performance visualizations
│   └── metrics/                 # Detailed performance metrics
├── logs/                        # Training logs
│   └── tensorboard/             # TensorBoard logs
└── checkpoints/                  # Training checkpoints

---

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/RL-for-LDI-Project.git](https://github.com/your-username/RL-for-LDI-Project.git)
    cd RL-for-LDI-Project
    ```
2.  **Install dependencies:**
    * If you don't have `nbconvert`, install it to convert Jupyter notebooks to `.py` scripts:
      ```bash
      pip install nbconvert
      ```
    * Once you have the `.py` files, install the rest of the dependencies from your `requirements.txt` file.
3.  **Prepare the data:**
    ```bash
    python src/data/processor.py
    ```
4.  **Train the model:**
    ```bash
    python src/training/train.py --algorithm PPO --timesteps 1000000
    ```
5.  **Run backtesting and evaluation:**
    ```bash
    python src/evaluation/backtest.py --model_path models/best_model/best_model --n_episodes 100
    ```

---

## Team and Acknowledgments

* **Authors**: Santiago Figueiras, Roberto Torres, and Diego Octavio Pérez
* **Acknowledgments**: Developed as part of the graduation requirements for the Financial Engineering program at **ITESO, Universidad Jesuita de Guadalajara**.
