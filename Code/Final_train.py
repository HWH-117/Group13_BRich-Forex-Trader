#!/usr/bin/env python
# coding: utf-8

# In[ ]:
"""
BRich Forex Trader - Training Script - Final Project Group 13
"""
#%%
# =============================================================================
# Step 1 Import Libraries
# =============================================================================
import os
import math
import argparse
import json
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.regression import MeanAbsoluteError
import joblib
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import warnings
import glob

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#%%
# =============================================================================
# Step 2 Configuration Management Parameters
# =============================================================================
CFG = {
    # --- Data related settings ---
    "symbols_config": {
        "EURUSD=X": {"inverse": True, "name": "USDEUR=X"}, 
        "GBPUSD=X": {"inverse": True, "name": "USDGBP=X"}, 
        "USDJPY=X": {"inverse": False, "name": "USDJPY=X"},
    },
    "market_symbol": "SPY",         # Market benchmark index (S&P 500)
    "start": "2014-01-01",          # Start date
    "end": "2025-06-15",            # End date
    "interval": "1d",               # Daily data

    # --- Model and training related settings ---
    "seq_len": 60,                  # Length of the input sequence for the model (looking at past 60 days of data)
    "batch_size": 64,               
    "lr": 3e-4,                     # Learning Rate
    "epochs": 100,                  
    "device": "cuda" if torch.cuda.is_available() else "cpu", # use GPU
    "num_workers": 0,               
    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "Training_outputs"),  # Save in same directory as script

    # --- Backtesting and trading strategy settings ---
    "simulation": {
        "days": 90,                                     # Number of days for backtesting simulation
        "initial_capital_per_asset": 10000.0,           # Initial capital for each asset
        "leverage": 5,                                  # Leverage multiplier
        "trade_size_lots": 1,                            
        "entry_atr_threshold": 0.5,                     # Entry threshold (enter when predicted volatility > 0.5 * ATR)
        "stop_loss_atr_multiplier": 1.5,                # Stop loss multiplier (set stop loss at 1.5 * ATR)
    }
}
os.makedirs(CFG["output_dir"], exist_ok=True)

#%%
# =============================================================================
# Step 3 Data Processing Functions
# =============================================================================

def download_and_process(sym: str, config: dict) -> pd.DataFrame:
    """
    Download historical data for the specified financial instrument from Yahoo Finance and perform initial cleaning and formatting.
    1. Use yfinance to download data.
    2. Convert all column names to lowercase.
    3. Check if necessary columns exist and ensure the 'volume' column is present.
    4. Remove rows with null values and filter out invalid data where price is less than or equal to 0.
    5. Add 'symbol' column and convert 'date' column to datetime format.
    """
    df = yf.download(sym, start=CFG["start"], end=CFG["end"], interval=CFG["interval"], progress=False, auto_adjust=False).reset_index()
    if df.empty: return pd.DataFrame()
    df.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in df.columns]
    required_cols = {'date', 'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Data {sym} is missing necessary columns: {required_cols - set(df.columns)}")
    if 'volume' not in df.columns:
        df['volume'] = 0
    df = df[list(required_cols | {'volume'})].copy()
    df.dropna(inplace=True); df = df[df["close"] > 0]
    if config.get("inverse", False):
        df[['open', 'high', 'low', 'close']] = 1.0 / df[['open', 'high', 'low', 'close']]
        df.rename(columns={'high': 'temp_low', 'low': 'high'}, inplace=True)
        df.rename(columns={'temp_low': 'low'}, inplace=True)
    df["symbol"] = config["name"]; df['date'] = pd.to_datetime(df['date'])
    return df

def add_indicators(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical analysis indicators and features required by the model to the given DataFrame.
    1. Set the 'date' column as the index for time series indicator calculations.
    2. Use the pandas_ta package to calculate EMA, RSI, ATR, Bollinger Bands, StochRSI, ROC, MACD, Ichimoku, etc.
    3. Calculate the rolling volatility of prices.
    4. Restore the index and add date-related features (day of the week, month).
    5. Merge with market data (SPY) to calculate market correlation `market_corr`.
    6. Safely fill null values in `market_corr` (fill with 0).
    7. Perform forward and backward filling on the entire DataFrame and remove final null values to ensure clean data.
    """
    df = df.copy()
    df.set_index('date', inplace=True)
    
    # Calculate technical indicators
    df.ta.ema(length=5, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.stochrsi(length=14, append=True)
    df.ta.roc(length=10, append=True)
    df.ta.macd(append=True)
    df['volatility'] = df['close'].rolling(window=20).std()
    df.ta.ichimoku(append=True)
    
    # Reset index and add date features
    df.reset_index(inplace=True)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Calculate market correlation
    merged = pd.merge(df, market_df[['date', 'close']], on='date', how='left', suffixes=('', '_market'))
    merged['market_corr'] = merged['close'].rolling(window=20).corr(merged['close_market'])
    df['market_corr'] = merged['market_corr']
    
    # Handle infinite values and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values
    df['market_corr'] = df['market_corr'].fillna(0)
    
    # Forward fill, backward fill, and drop any remaining NaN
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    # Additional validation to ensure no infinite values remain
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].mean())
    
    return df.reset_index(drop=True)

def prepare_data(mode='train'):
    """
    Complete data preparation process, from downloading to final formatting, to prepare for model training or prediction.
    1. Download and process all configured currency pairs and market data.
    2. Add technical indicators to each currency pair's data.
    3. Merge all data into a large DataFrame.
    4. Check if the data length is sufficient for backtesting split; if not, raise an error.
    5. Split the dataset into training and backtesting sets.
    6. Standardize the features of the training set (StandardScaler) and save the Scaler.
    7. Calculate the target variable "log return".
    8. Construct the model's input sequence (X), target sequence (Y), and asset identifiers (S).
    9. Save all necessary data processing objects (Scaler, feature list) for use during prediction.
    """
    all_symbols_config = {**CFG["symbols_config"], CFG["market_symbol"]: {"inverse": False, "name": CFG["market_symbol"]}}
    raw_dfs = {conf['name']: download_and_process(sym, conf) for sym, conf in all_symbols_config.items()}
    market_df = raw_dfs.pop(CFG["market_symbol"])
    all_dfs = [add_indicators(df, market_df) for df in raw_dfs.values()]
    full_df = pd.concat(all_dfs).sort_values(["symbol", "date"]).reset_index(drop=True)

    if mode == 'predict': return full_df, raw_dfs

    unique_dates = sorted(full_df['date'].unique())
    
    required_data_length = CFG['simulation']['days'] + CFG['seq_len'] + 5
    if len(unique_dates) < required_data_length:
        raise ValueError(
            f"Insufficient data for backtesting split.\n"
            f"Available days after indicator calculation: {len(unique_dates)} days.\n"
            f"But the program requires at least: {required_data_length} days ({CFG['seq_len']} days of historical data + {CFG['simulation']['days']} days of backtesting + buffer).\n"
            f"Solution: Please set the 'start' date in CFG earlier (e.g., '2014-01-01'), or reduce the 'days' in 'simulation'."
        )
    
    split_date = unique_dates[-CFG['simulation']['days']]
    train_df = full_df[full_df['date'] < split_date].copy()

    feature_cols = [c for c in full_df.columns if c not in ['date', 'symbol', 'close', 'log_return']]
    feature_scalers = {}
    present_symbols = train_df['symbol'].unique()

    for sym in present_symbols:
        m_train = train_df['symbol'] == sym
        train_subset = train_df.loc[m_train, feature_cols].dropna()
        
        # Additional validation before scaling
        if not train_subset.empty:
            # Replace any remaining infinite values with NaN
            train_subset = train_subset.replace([np.inf, -np.inf], np.nan)
            # Fill NaN with mean values
            train_subset = train_subset.fillna(train_subset.mean())
            # Verify no infinite values remain
            if not np.isfinite(train_subset.values).all():
                print(f"Warning: Infinite values found in {sym} data. Attempting to clean...")
                train_subset = train_subset.replace([np.inf, -np.inf], np.nan)
                train_subset = train_subset.fillna(train_subset.mean())
            
            feature_scalers[sym] = StandardScaler().fit(train_subset)

    X, Y, S = [], [], []
    symbol_to_idx = {name: i for i, name in enumerate(present_symbols)}

    for sym in present_symbols:
        if sym not in feature_scalers: continue
        
        m_sym = train_df['symbol'] == sym
        sym_df_full = train_df[m_sym].copy()
        sym_df_full['log_return'] = np.log(sym_df_full['close'].shift(-1) / sym_df_full['close'])
        sym_df_full.dropna(subset=['log_return'], inplace=True)

        sym_df_scaled = sym_df_full.copy()
        
        valid_feature_cols = [col for col in feature_cols if col in sym_df_scaled.columns]
        sym_df_scaled[valid_feature_cols] = feature_scalers[sym].transform(sym_df_scaled[valid_feature_cols])

        for i in range(CFG["seq_len"], len(sym_df_scaled)):
            seq = sym_df_scaled.iloc[i-CFG["seq_len"]:i][valid_feature_cols].values
            target_log_return = sym_df_full.iloc[i-1]['log_return']
            X.append(seq); Y.append(target_log_return); S.append(symbol_to_idx[sym])

    output_dir = os.path.abspath(CFG["output_dir"])
    joblib.dump(feature_scalers, os.path.join(output_dir, 'feature_scalers.pkl'))
    joblib.dump(valid_feature_cols, os.path.join(output_dir, 'feature_cols.pkl'))
    joblib.dump(present_symbols, os.path.join(output_dir, 'symbols.pkl'))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, 1), np.array(S), len(valid_feature_cols), len(present_symbols)

#%%
# =============================================================================
# Step 4 PyTorch Model Definition
# =============================================================================

# PyTorch Dataset class for wrapping Numpy arrays into a format usable by PyTorch datasets
class ForexDataset(Dataset):
    def __init__(self, X, Y, S): self.X, self.Y, self.S = X, Y, S
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.Y[i], dtype=torch.float32), torch.tensor(self.S[i], dtype=torch.long)

# Multi-scale 1D convolution module to extract features at different time scales using different kernel sizes
class MSConv(nn.Module):
    def __init__(self, i, o=32, k=(3,5,7)): super().__init__(); self.convs=nn.ModuleList([nn.Conv1d(i,o,j,padding=j//2)for j in k]); self.bn=nn.BatchNorm1d(o*len(k))
    def forward(self, x): return F.gelu(self.bn(torch.cat([c(x) for c in self.convs], dim=1)))

# Positional encoding module to provide position information for each time step in the sequence to the Transformer
class PosEnc(nn.Module):
    def __init__(self, d, max_len=512): super().__init__(); pe=torch.zeros(max_len,d); p=torch.arange(0,max_len)[:,None]; div=torch.exp(torch.arange(0,d,2)*(-math.log(10000.0)/d)); pe[:,0::2]=torch.sin(p*div); pe[:,1::2]=torch.cos(p*div); self.register_buffer("pe",pe)
    def forward(self, x): return x + self.pe[:x.size(1)]

# Module combining multi-scale convolution and Transformer, first using CNN to extract local features, then using Transformer to capture global dependencies
class MSConvTrans(nn.Module):
    def __init__(self, nf, T, d=128,h=4,nl=3): super().__init__(); self.ms=MSConv(nf);self.proj=nn.Conv1d(96,d,1);self.pos=PosEnc(d,T);el=nn.TransformerEncoderLayer(d,h,d*4,batch_first=True,norm_first=True,dropout=0.15);self.enc=nn.TransformerEncoder(el,nl)
    def forward(self,x): return self.enc(self.pos(self.proj(self.ms(x.permute(0,2,1))).permute(0,2,1)))

# Hybrid backbone network combining Transformer and GRU, ultimately outputting complete time series features
class HybridBackbone(nn.Module):
    def __init__(self, nf, T, d_t=128, d_g=128, n_g=2):
        super().__init__()
        self.transformer=MSConvTrans(nf,T,d=d_t)
        self.gru=nn.GRU(d_t,d_g,n_g,batch_first=True,bidirectional=True,dropout=0.25 if n_g>1 else 0)
        self.out_dim=d_g*2
    def forward(self, x):
        gru_out, _ = self.gru(self.transformer(x))
        return gru_out

# Attention mechanism module allowing the model to give different attention to different parts of the sequence
class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * features, dim=1)
        return context_vector

# PyTorch Lightning main model integrating all modules and defining the complete training and validation process
class SupTask(pl.LightningModule):
    def __init__(self, n_features, seq_len, n_symbols, lr):
        super().__init__()
        self.save_hyperparameters() # Automatically save hyperparameters
        self.backbone=HybridBackbone(n_features,seq_len) 
        self.sym_emb=nn.Embedding(n_symbols,16) # Asset embedding layer, learning unique properties of each currency pair
        self.attention_head = AttentionHead(self.backbone.out_dim, 128) 
        flat_dim=self.backbone.out_dim + 16 # Dimension of concatenated features
        self.head=nn.Sequential(nn.Linear(flat_dim,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.6),nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(0.6))
        self.reg_head=nn.Linear(256,1) # Final regression output layer
        self.loss_fn=nn.MSELoss() # Loss function: Mean Squared Error
        self.mae=MeanAbsoluteError() # Evaluation metric: Mean Absolute Error

    # Define the forward propagation path of the model
    def forward(self, x, s):
        seq_features = self.backbone(x)
        context_vector = self.attention_head(seq_features)
        h=torch.cat([context_vector, self.sym_emb(s)], 1)
        return self.reg_head(self.head(h))

    # Define a single training step
    def training_step(self, b, _):
        x,y,s=b; y_hat=self(x,s); loss=self.loss_fn(y_hat,y)
        self.log('tr_loss',loss,on_epoch=True,prog_bar=False,logger=True)
        self.log('tr_mae',self.mae(y_hat,y),on_epoch=True, prog_bar=False, logger=True)
        return loss

    # Define a single validation step
    def validation_step(self, b, _):
        x,y,s=b; y_hat=self(x,s); loss=self.loss_fn(y_hat,y); mae=self.mae(y_hat,y)
        self.log_dict({'va_loss':loss,'va_mae':mae}, prog_bar=True, on_epoch=True, logger=True)
        return loss

    # Configure optimizer and learning rate scheduler
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "va_loss"}}

#%%
# =============================================================================
# Step 5 Trading Simulation Related Classes and Functions
# =============================================================================

# Rule-based trading decision model
class TradingActionModel:
    def get_action(self, predicted_log_return, current_atr, atr_threshold, current_price):
        """Decide trading action (LONG, SHORT, HOLD) based on predicted log return and ATR"""
        predicted_price_change = (math.exp(predicted_log_return) - 1) * current_price
        atr_move_threshold = atr_threshold * current_atr
        if predicted_price_change > atr_move_threshold: return 'LONG'
        elif predicted_price_change < -atr_move_threshold: return 'SHORT'
        else: return 'HOLD'

# Main trading simulator class
class TradingSimulator:
    def __init__(self):
        """Initialize the simulator, load model and data, set initial capital and positions"""
        print("--- Initializing BRich Trading Simulator ---")
        self.model, self.feature_scalers, self.feature_cols, self.symbols = self._load_artifacts()
        self.action_model = TradingActionModel()
        self.full_history_df, _ = prepare_data(mode='predict')
        self.sim_symbols = self.symbols
        self.symbol_to_idx = {name: i for i, name in enumerate(self.symbols)}
        self.initial_capital = {sym: CFG['simulation']['initial_capital_per_asset'] for sym in self.symbols}
        self.capital = self.initial_capital.copy()
        self.positions = {sym: {'status': 'FLAT', 'entry_price': 0, 'stop_loss': 0, 'size': 0} for sym in self.symbols}
        self.daily_records = []
        self.prediction_records = []

    def _load_artifacts(self):
        """Load trained model, Scaler, feature list, and other auxiliary files from disk"""
        output_dir = CFG["output_dir"]
        feature_cols = joblib.load(os.path.join(output_dir, 'feature_cols.pkl'))
        n_features = len(feature_cols)
        n_symbols = len(joblib.load(os.path.join(output_dir, 'symbols.pkl')))
        model_path = os.path.join(output_dir, 'best-model.ckpt')
        model = SupTask.load_from_checkpoint(model_path, n_features=n_features, seq_len=CFG['seq_len'], n_symbols=n_symbols, lr=CFG['lr'])
        feature_scalers = joblib.load(os.path.join(output_dir, 'feature_scalers.pkl'))
        symbols = joblib.load(os.path.join(output_dir, 'symbols.pkl'))
        print(f"✅ Model and artifacts loaded from '{output_dir}'.")
        return model.to(CFG['device']).eval(), feature_scalers, feature_cols, symbols

    def _get_prediction(self, sym, current_date):
        """
        Function: Generate price prediction for the specified product and date.
        Steps:
        1. Extract historical data before that date as input sequence.
        2. Standardize features using the corresponding Scaler.
        3. Convert data to PyTorch tensor.
        4. Input tensor into the model to obtain predicted log return.
        5. Convert log return back to predicted absolute price.
        """
        past_data = self.full_history_df[(self.full_history_df['date'] < current_date) & (self.full_history_df['symbol'] == sym)]
        if len(past_data) < CFG['seq_len']: return None, None
        seq_df = past_data.tail(CFG['seq_len']).copy()
        current_price = seq_df.iloc[-1]['close']
        
        valid_feature_cols = [col for col in self.feature_cols if col in seq_df.columns]
        seq_df[valid_feature_cols] = self.feature_scalers[sym].transform(seq_df[valid_feature_cols])
        
        seq_tensor = torch.tensor(seq_df[valid_feature_cols].values, dtype=torch.float32).unsqueeze(0).to(CFG['device'])
        sym_tensor = torch.tensor([self.symbol_to_idx[sym]], dtype=torch.long).to(CFG['device'])
        
        with torch.no_grad():
            predicted_log_return = self.model(seq_tensor, sym_tensor).item()
        predicted_price = current_price * math.exp(predicted_log_return)
        return predicted_price, predicted_log_return

    def run(self):
        """
        Function: Execute the main backtesting loop.
        Steps:
        1. Iterate over the specified backtesting days.
        2. For each day, perform operations for each currency pair.
        3. Calculate the floating P&L of the current position and check for stop loss triggers.
        4. Get the latest predictions from the model.
        5. Based on predictions and trading rules, decide whether to open, close, or hold a position.
        6. Update position and capital status.
        7. Record all trading activities and financial status for the day.
        8. After the loop, calculate and report final performance.
        """
        print(f"\n--- 🚀 Starting BRich Trading Simulation ({CFG['simulation']['days']} days) ---")
        unique_dates = sorted(self.full_history_df['date'].unique())
        test_dates = unique_dates[-CFG['simulation']['days']:]
        for date in test_dates:
            floating_pnl_today = 0
            for sym in self.sim_symbols:
                current_data = self.full_history_df[(self.full_history_df['date'] == date) & (self.full_history_df['symbol'] == sym)]
                if current_data.empty: continue
                current_price = current_data.iloc[0]['close']
                current_atr = current_data.iloc[0]['ATRr_14']
                pos = self.positions[sym]
                floating_pnl_sym = 0
                if pos['status'] != 'FLAT':
                    price_diff = current_price - pos['entry_price']
                    pnl = price_diff if pos['status'] == 'LONG' else -price_diff
                    floating_pnl_sym = pnl * pos['size']
                    if (pos['status'] == 'LONG' and current_price < pos['stop_loss']) or \
                       (pos['status'] == 'SHORT' and current_price > pos['stop_loss']):
                        self.capital[sym] += floating_pnl_sym
                        print(f"🛑 STOP-LOSS triggered for {sym} on {date.date()}: Realized PnL: ${floating_pnl_sym:.2f}")
                        pos['status'] = 'FLAT'; pos['size'] = 0; floating_pnl_sym = 0
                floating_pnl_today += floating_pnl_sym
                predicted_price, pred_log_return = self._get_prediction(sym, date)
                if pred_log_return is None:
                    action = 'HOLD'
                else:
                    action = self.action_model.get_action(pred_log_return, current_atr, CFG['simulation']['entry_atr_threshold'], current_price)
                if pos['status'] == 'FLAT':
                    if action == 'LONG' or action == 'SHORT':
                        pos['status'] = action
                        pos['entry_price'] = current_price
                        pos['size'] = CFG['simulation']['trade_size_lots']
                        sl_multiplier = CFG['simulation']['stop_loss_atr_multiplier']
                        if action == 'LONG': pos['stop_loss'] = current_price - sl_multiplier * current_atr
                        else: pos['stop_loss'] = current_price + sl_multiplier * current_atr
                        print(f"🚀 ENTER {action} {sym} at ${current_price:.4f} on {date.date()}")
                elif (pos['status'] == 'LONG' and action == 'SHORT') or \
                     (pos['status'] == 'SHORT' and action == 'LONG'):
                    self.capital[sym] += floating_pnl_sym
                    print(f"💰 CLOSE {pos['status']} {sym} on {date.date()}: Realized PnL: ${floating_pnl_sym:.2f}")
                    pos['status'] = 'FLAT'; pos['size'] = 0; floating_pnl_sym = 0
                self.daily_records.append({
                    'Date': date.strftime('%Y-%m-%d'), 'Symbol': sym, 'Actual Price': current_price,
                    'Predicted Price': predicted_price, 'Action': action, 'Position Status': pos['status'],
                    'Floating PnL': floating_pnl_sym, 'Total Equity': self.capital[sym] + floating_pnl_today
                })
                if predicted_price is not None and current_price is not None:
                    self.prediction_records.append({
                        'date': date,
                        'symbol': sym,
                        'actual': current_price,
                        'predicted': predicted_price
                    })
        print("✅ Simulation completed!")
        self.calculate_and_show_roi()
        self.export_results()
        self._calculate_and_print_accuracy()
        return pd.DataFrame(self.daily_records)

    def calculate_and_show_roi(self):
        """Calculate and print the final performance summary, including ROI"""
        final_capital = sum(self.capital.values())  # Sum all symbol capitals
        if self.daily_records:
            last_day_pnl = sum(rec['Floating PnL'] for rec in self.daily_records if rec['Date'] == self.daily_records[-1]['Date'])
            final_capital += last_day_pnl
        initial_total = sum(self.initial_capital.values())  # Sum all initial capitals
        roi = ((final_capital - initial_total) / initial_total) * 100
        print("\n--- 📈 Performance Summary ---")
        print(f"Initial Capital: ${initial_total:,.2f}")
        print(f"Final Equity:   ${final_capital:,.2f}")
        print(f"Return on Investment (ROI): {roi:.2f}%")
        print("---------------------------\n")

    def export_results(self):
        """Export simulation results to CSV and JSON files for further analysis or GUI use"""
        if not self.daily_records:
            print("⚠️ No records to export.")
            return

        # Build DataFrame of new trades
        df_new = pd.DataFrame(self.daily_records)
        output_dir = CFG["output_dir"]
        csv_path = os.path.join(output_dir, "trading_log.csv")

        # If an existing log exists, read and append new entries
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Remove any exact duplicates (same date & symbol)
            df_combined = df_combined.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
        else:
            df_combined = df_new

        # Save the combined log
        df_combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ Trading records exported to: {csv_path}")

        # Also update JSON for the GUI
        json_path = os.path.join(output_dir, "trading_log.json")
        df_combined.to_json(
            json_path,
            orient='records',
            indent=4,
            date_format='iso',
            force_ascii=False
        )
        print(f"✅ GUI exported trading records to: {json_path}")

    def _calculate_and_print_accuracy(self):
        """
        Calculate the model's prediction accuracy metrics (MAE, MAPE).
        """
        if not self.prediction_records:
            print("\nNo prediction records available for accuracy analysis.")
            return
        records_df = pd.DataFrame(self.prediction_records)
        print("\n--- 🎯 Model Prediction Accuracy Analysis ---")
        mae = np.mean(np.abs(records_df['predicted'] - records_df['actual']))
        mape = np.mean(np.abs((records_df['predicted'] - records_df['actual']) / records_df['actual'])) * 100
        print(f"Overall Accuracy:\n - Mean Absolute Error (MAE): {mae:.5f}\n - Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Calculate accuracy for each prediction
        records_df['accuracy'] = 100 * (1 - np.abs((records_df['predicted'] - records_df['actual']) / records_df['actual']))
        
        # Save accuracy data to CSV for GUI
        accuracy_path = os.path.join(CFG["output_dir"], "prediction_accuracy.csv")
        records_df.to_csv(accuracy_path, index=False)
        print(f"✅ Accuracy data saved to: {accuracy_path}")

#%%
# =============================================================================
# Step 6 Main Execution Process and Visualization
# =============================================================================

def plot_training_history(log_dir):
    try:
        event_files = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events.out.tfevents" in f])
        if not event_files: print("❌ Error: No TensorBoard event files found."); return
        event_file = event_files[-1]
        print(f"📄 Reading training logs from '{event_file}'...")
        ea = event_accumulator.EventAccumulator(event_file, size_guidance={'scalars': 0})
        ea.Reload()
        available_tags = ea.Tags()['scalars']
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        fig.suptitle("Training & Validation History", fontsize=16, fontweight='bold')
        
        # Get all data first
        tr_loss = pd.DataFrame(ea.Scalars('tr_loss_epoch')) if 'tr_loss_epoch' in available_tags else None
        va_loss = pd.DataFrame(ea.Scalars('va_loss'))         if 'va_loss'         in available_tags else None
        tr_mae  = pd.DataFrame(ea.Scalars('tr_mae_epoch'))   if 'tr_mae_epoch'    in available_tags else None
        va_mae  = pd.DataFrame(ea.Scalars('va_mae'))         if 'va_mae'          in available_tags else None

        # Assign a simple 1-based epoch index to each row
        if tr_loss is not None:
            tr_loss['epoch'] = np.arange(len(tr_loss)) + 1
        if va_loss is not None:
            va_loss['epoch'] = np.arange(len(va_loss)) + 1
        if tr_mae is not None:
            tr_mae['epoch'] = np.arange(len(tr_mae)) + 1
        if va_mae is not None:
            va_mae['epoch'] = np.arange(len(va_mae)) + 1

        # Plot Loss graph
        if tr_loss is not None and va_loss is not None:
            axes[0].plot(tr_loss['epoch'], tr_loss['value'], label='Training Loss', color='deepskyblue', marker='.')
            axes[0].plot(va_loss['epoch'], va_loss['value'], label='Validation Loss', color='orangered',    marker='.')
            axes[0].set_title('Model Loss', fontsize=14)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)
            # Scale x-axis to actual logged epochs
            max_e = max(tr_loss['epoch'].max(), va_loss['epoch'].max())
            axes[0].set_xlim(1, max_e)
        else:
            axes[0].text(0.5, 0.5, 'Loss data not available.', ha='center', va='center')
            axes[0].set_title('Model Loss', fontsize=14)

        # Plot MAE graph
        if tr_mae is not None and va_mae is not None:
            axes[1].plot(tr_mae['epoch'], tr_mae['value'], label='Training MAE',              color='deepskyblue', marker='.')
            axes[1].plot(va_mae['epoch'], va_mae['value'], label='Validation MAE',            color='orangered',    marker='.')
            axes[1].set_title('Model Mean Absolute Error (MAE)', fontsize=14)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)
            # Scale x-axis to actual logged epochs
            max_e = max(tr_mae['epoch'].max(), va_mae['epoch'].max())
            axes[1].set_xlim(1, max_e)
        else:
            axes[1].text(0.5, 0.5, 'MAE data not available.', ha='center', va='center')
            axes[1].set_title('Model Mean Absolute Error (MAE)', fontsize=14)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(CFG["output_dir"], "training_history.png")
        plt.savefig(plot_path)
        print(f"✅ Training history plot saved to: {plot_path}")
        plt.show()
    except Exception as e:
        print(f"❌ Error occurred while plotting training history: {e}")

def stage_train():
    print("\n--- 🚀 Starting Training Phase ---")
    X, Y, S, n_features, n_symbols = prepare_data('train')
    if n_features == 0 or n_symbols == 0 or len(X) == 0:
        print("❌ Error: No valid training samples generated after data preparation, cannot continue training.")
        return
    
    # Split training and validation sets
    split_idx = int(len(X) * 0.9)
    ds_tr = ForexDataset(X[:split_idx], Y[:split_idx], S[:split_idx])
    ds_va = ForexDataset(X[split_idx:], Y[split_idx:], S[split_idx:])
    
    # Create DataLoader
    dl_tr = DataLoader(ds_tr, batch_size=CFG["batch_size"], shuffle=True, num_workers=CFG["num_workers"])
    dl_va = DataLoader(ds_va, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])
    
    # Initialize model and logger
    model = SupTask(n_features=n_features, seq_len=CFG["seq_len"], n_symbols=n_symbols, lr=CFG['lr'])
    logger = TensorBoardLogger(save_dir=CFG["output_dir"], name=None, default_hp_metric=False, version=0)
    early_stop_cb = EarlyStopping(monitor="va_mae", patience=20, mode="min", verbose=True)

    # Add custom progress callback
    class ProgressCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started")
        
        def on_train_epoch_end(self, trainer, pl_module):
            print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} completed")

    # Initialize Trainer
    trainer = pl.Trainer(max_epochs=CFG["epochs"], accelerator=CFG["device"], 
                         callbacks=[early_stop_cb, ProgressCallback()], logger=logger, 
                         log_every_n_steps=10, enable_progress_bar=False, 
                         enable_model_summary=False, enable_checkpointing=False)
    trainer.fit(model, train_dataloaders=dl_tr, val_dataloaders=dl_va)

    # Manually save a single checkpoint with a fixed name
    ckpt_path = os.path.join(CFG["output_dir"], "best-model.ckpt")
    trainer.save_checkpoint(ckpt_path)
    CFG['model_path'] = ckpt_path
    print(f"\n✅ Training completed! Model saved to: '{ckpt_path}'")

    # Plot training history
    plot_training_history(logger.log_dir)

def stage_predict():
    """Execute prediction and backtesting simulation phase"""
    print("\n--- 🚀 Starting Prediction and Backtesting Phase ---")
    model_path = os.path.join(CFG["output_dir"], 'best-model.ckpt')
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found in '{CFG['output_dir']}'.")
        return
    simulator = TradingSimulator()
    simulator.run()

#%%
# Main entry point of the Python script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BRich Forex Trader (Final)")
    parser.add_argument("--stage", type=str, default="train_and_predict", choices=["train", "predict", "train_and_predict"])
    parser.add_argument("--currency", type=str, default=None, help="Currency pair to train (e.g. EUR/USD)")
    parser.add_argument("--epochs", type=int, default=CFG["epochs"], help="Number of training epochs")
    parser.add_argument("--leverage", type=int, default=CFG["simulation"]["leverage"], help="Leverage for trading")
    parser.add_argument("--initial_capital", type=int, default=CFG["simulation"]["initial_capital_per_asset"], help="Initial capital per asset")
    parser.add_argument("--start_date", type=str, help="Start date for training data")
    parser.add_argument("--end_date", type=str, help="End date for training data")
    args, _ = parser.parse_known_args()

    # ------------------ PRESERVE PREVIOUSLY TRAINED CURRENCIES ------------------
    # Load any symbols.pkl you generated last time, and merge into your config
    symbols_pkl = os.path.join(CFG["output_dir"], "symbols.pkl")
    if os.path.exists(symbols_pkl):
        prev_syms = joblib.load(symbols_pkl)
        for sym in prev_syms:
            if sym not in CFG["symbols_config"]:
                CFG["symbols_config"][sym] = {"inverse": False, "name": sym}
    # ---------------------------------------------------------------------------

    # ------------------ INCLUDE NEW GUI CURRENCY (if any) ----------------------
    if args.currency:
        # Normalize GUI format "EUR/USD" → yfinance key "EURUSD=X"
        sym_key = args.currency.replace("/", "").upper() + "=X"
        if sym_key not in CFG["symbols_config"]:
            CFG["symbols_config"][sym_key] = {"inverse": False, "name": sym_key}
        print(f"🔍 Included currency in training & simulation: {sym_key}")
    # ---------------------------------------------------------------------------

    # Apply other overrides (epochs, dates, leverage, initial capital)…
    CFG["epochs"]                          = args.epochs
    if args.start_date:   CFG["start"]     = args.start_date
    if args.end_date:     CFG["end"]       = args.end_date
    if args.leverage:     CFG["simulation"]["leverage"]                  = args.leverage
    if args.initial_capital: CFG["simulation"]["initial_capital_per_asset"] = args.initial_capital

    # Now when you call stage_train()/stage_predict(), 
    # prepare_data() will see all the old + new currencies.
    if args.stage == "train":
        stage_train()
    elif args.stage == "predict":
        stage_predict()
    elif args.stage == "train_and_predict":
        stage_train()
        stage_predict()

# End of Code