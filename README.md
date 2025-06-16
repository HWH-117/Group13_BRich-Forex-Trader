![{A4C07259-B6C6-4F02-B398-B4B4BD1749C8}](https://github.com/user-attachments/assets/69d66efc-d544-4ff9-b1cf-d33c800493c5)

## BRich AI Forex Simulator!

**1. Project Overview**

 This is an AI-powered foreign exchange (FX) trading simulation system. The project uses a sophisticated deep learning model to predict market trends and executes trades in a realistic backtesting environment.

  * **AI Model:** Utilizes a hybrid Transformer, GRU, and Attention Mechanism architecture to predict the daily price movements (log returns) of currency pairs.

  * **Backtesting Engine:** Simulates trading for USD/JPY, USD/EUR, and USD/GBP over a 90-day period, incorporating essential concepts like leverage, floating PnL, and stop-loss.
  * **Interactive GUI Dashboard:** A user-friendly interface built with Streamlit to manage model training, simulation, and results analysis.


**2. Directory Structure**

```text
Group_13_BRich Forex trader/
│
├── BRich_APP/                  # GUI Application Files
│   ├── BRich_GUI.py            # [GUI] Main Streamlit application script
│   ├── requirements.txt        # Python dependencies for the GUI
│   ├── fx_background.png       # Background image
│   └── ... (Other image assets)
│
├── Training_outputs/     # [IMPORTANT] All training and simulation outputs are saved here
│   ├── best-model.ckpt         # Trained AI model weights 
│   ├── equity_curve.json       # [GUI CORE] Raw data for the equity curve chart 
│   ├── simulation_details.csv  # [GUI CORE] Detailed daily simulation records 
│   ├── trading_log.csv         # [GUI CORE] Trade logs
│   └── ... (Other .pkl) 
│
├── Final_simulate.py           # [Executable] Main script for running backtest simulations
├── Final_train.py              # [Executable] Main script for training the AI model
├── requirements.txt            # Python dependencies for the train & simulate
└── ... (Other notebooks and log files)
```



**3. Quick Start (Setup & Run)**

Setup
Prerequisites: Python 3.9.21+ and pip.

Install Dependencies: Create a requirements.txt file with the content below and run the command:

**pip install -r requirements.txt**

requirements.txt content

* numpy

* pandas

* yfinance

* pandas-ta

* torch

* pytorch-lightning

* scikit-learn

* matplotlib

* tensorboard

* openpyxl


**How to Run**

**There are two primary ways to run the project for evaluation:**

* **A) Run Simulation with Pre-trained Model:** This uses the included **best-model.ckpt** to generate the latest results.

* **B) Run in TA Evaluation Mode (with external Excel files):** Place the TA-provided **fx_data.xlsx** and **fake_fx_data.xlsx** in the project's root folder. The script will automatically detect them and run in TA mode.

* **(Optional) Re-train the AI Model:** To train the model from scratch, run the training script. This will overwrite the existing best-model.ckpt.

The GUI provides the easiest way to interact with the project. You can perform all operations from the web interface.

Launch the Application:
In your terminal, run the following command:

**streamlit run BRich_APP/BRich_GUI.py**

Use the Interface:
Your browser will automatically open the dashboard. Use the sidebar to navigate between pages:

* **Forex Analysis:** Download market data, train new models, and view training charts.
* **Trading Simulation:** Run back-tests with your trained model and view detailed equity curves and trade analysis charts.
* **Summary:** Review consolidated performance metrics like ROI, MAE, and MAPE for both training and simulation.




**4. Key Output Files for GUI & Evaluation**
All results are saved in the BRich_Training_outputs/ directory. The most important files for the GUI team and the TA are:

* **simulation_details.json:**
Content: The core data file. Contains a detailed log for each day of the simulation, including date, symbol, prices, action taken, and PnL.
Usage: The primary data source for populating the GUI's main trading log table.

* **equity_curve.json:**
Content: Daily timeseries data of the total account equity.
Usage: For drawing an interactive equity curve chart in the GUI.

* **Image Files (.png):**
equity_curve.png: A static chart showing the overall profit/loss over time.
trading_simulation_analysis.png: Detailed charts showing trade entry and exit points against market prices.
Usage: Can be directly loaded and displayed in the GUI for a quick visual summary.



##  Author

**HWH**  
Master Student in International Master Program on Intelligent Manufacturing, NCKU  
Email: [Megamind11129@gamil.com]




