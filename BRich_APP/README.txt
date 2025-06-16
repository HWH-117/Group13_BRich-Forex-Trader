Group 13 – BRich Forex Trader GUI
=================================

Welcome to the BRich Forex Trader Dashboard, a Streamlit-based interface to train AI models, run back-tests, and analyze trading performance for various currency pairs.

Prerequisites
-------------
- Python 3.9 or higher
- pip package manager

Installation
------------
1. Open a terminal and navigate to the GUI directory:
   ```bash
   cd "BRich_APP"
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows PowerShell
   .\venv\Scripts\Activate
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Running the GUI
---------------
Launch the Streamlit application:
```bash
streamlit run BRich_GUI.py
```
Use the sidebar to navigate between pages:
- **Landing Page**: Overview and quick navigation.
- **Forex Analysis**: Download market data, train models, and view training charts.
- **Trading Simulation**: Run back-tests with your trained model; view equity curves and trade analysis.
- **Summary**: Consolidated performance metrics (ROI, MAE, MAPE) for training and simulation.

Related CLI Scripts
-------------------
You can also train or simulate directly via command-line:

- **Training** (`Final_train.py`):
  ```bash
  python Final_train.py --stage train --currency EUR/USD --epochs 100
  ```

- **Simulation** (`Final_simulate.py`):
  ```bash
  python Final_simulate.py --simulation_days 90 --currency EURUSD=X --leverage 5 --initial_capital 10000
  ```

Data Outputs
------------
All models, logs, and chart data are saved under `Training_outputs`:
- `best-model.ckpt`: Trained model checkpoint.
- `trading_log.json` / `.csv`: Detailed trade logs for the GUI.
- `simulation_details.json` / `.csv`: Daily simulation records.
- `equity_curve.json` / `.png`: Equity curve data and static chart.
- `training_history.png`: Static training history chart.

Support
-------
For issues or questions, consult inline code comments or contact the development team (Group 13).

