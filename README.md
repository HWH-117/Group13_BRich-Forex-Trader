
![{89228400-AB08-4962-B445-921484507A88}](https://github.com/user-attachments/assets/69f22630-db10-4c47-bf1d-cfcbd82818ed)

## BRich AI Forex Simulator!

1. Project Overview

 This is an AI-powered foreign exchange (FX) trading simulation system. The project uses a sophisticated deep learning model to predict market trends and executes trades in a realistic backtesting environment.

   AI Model Utilizes a hybrid Transformer, GRU, and Attention Mechanism architecture to predict the daily price movements (log returns) of currency pairs.

   Backtesting Engine Simulates trading for USDJPY, USDEUR, and USDGBP over a 90-day period, incorporating essential concepts like leverage, floating PnL, and stop-loss.


2. Directory Structure

```text
Group_13_BRich Forex trader
│
├── .ipynb_checkpoints         # (Jupyter Notebook checkpoints, can be ignored)
│
├── BRich_Training_outputs     # [IMPORTANT] All output files from training and simulation are saved here.
│   ├── best-model.ckpt             # The trained AI model weights.
│   ├── equity_curve.json           # [GUI CORE] Raw data for the equity curve chart.
│   ├── simulation_details.json     # [GUI CORE] Detailed daily trading records.
│   ├── equity_curve.png              # Generated equity curve chart image.
│   ├── trading_simulation_analysis.png # Generated trade analysis chart image.
│   ├── training_history.png        # Generated model training history image.
│   └── ... (and other helperlog files)
│
├── tb_logs                      # (TensorBoard logs, can be ignored for evaluation).
│
├── Final_predicted_simulation.ipynb  # Simulation script (Jupyter Notebook version).
├── Final_predicted_simulation.py     # [Executable] The main script for running backtest simulations.
├── Final_train.ipynb                 # Training script (Jupyter Notebook version).
└── Final_train.py                    # [Executable] The main script for training the AI model.
```



3. Quick Start (Setup & Run)

Setup
Prerequisites Python 3.9.21+ and pip.

Install Dependencies Create a requirements.txt file with the content below and run the command

pip install -r requirements.txt

requirements.txt content

 numpy

 pandas

 yfinance

 pandas-ta

 torch

 pytorch-lightning

 scikit-learn

 matplotlib

 tensorboard

 openpyxl


**How to Run**

There are two primary ways to run the project for evaluation

 * A) Run Simulation with Pre-trained Model This uses the included best-model.ckpt to generate the latest results.

 * B) Run in TA Evaluation Mode (with external Excel files) Place the TA-provided fx_data.xlsx and fake_fx_data.xlsx in the project's root folder. The script will automatically detect them and run in TA mode.

 * (Optional) Re-train the AI Model To train the model from scratch, run the training script. This will overwrite the existing best-model.ckpt.



4. Key Output Files for GUI & Evaluation
All results are saved in the BRich_Training_outputs directory. The most important files for the GUI team and the TA are

simulation_details.json
Content The core data file. Contains a detailed log for each day of the simulation, including date, symbol, prices, action taken, and PnL.
Usage The primary data source for populating the GUI's main trading log table.

equity_curve.json
Content Daily timeseries data of the total account equity.
Usage For drawing an interactive equity curve chart in the GUI.

Image Files (.png)
equity_curve.png A static chart showing the overall profitloss over time.
trading_simulation_analysis.png Detailed charts showing trade entry and exit points against market prices.
Usage Can be directly loaded and displayed in the GUI for a quick visual summary.



##  Author

HWH  
Master Student in International Master Program on Intelligent Manufacturing, NCKU  
Email [Megamind11129@gamil.com]




