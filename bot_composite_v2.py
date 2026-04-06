#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BOT TRADING COMPOSITE — VERSION 2 (APPRENTISSAGE TEMPS RÉEL)
═════════════════════════════════════════════════════════════
NOUVEAUTÉS v2 :
  ✔ Architecture double-thread : data (30s) + décision (2s)
      → L'IA prend une décision toutes les 2 secondes
      → Les données sont rafraîchies toutes les 30s (limite API)
  ✔ Apprentissage à CHAQUE PAS (plus uniquement à la clôture)
      → Récompense intermédiaire = variation de prix depuis la dernière step
      → L'IA ressent le mouvement du marché en temps réel
  ✔ Mémoire persistante complète
      → dqn_weights.pkl  : poids du réseau neuronal
      → dqn_memory.pkl   : replay buffer (transitions passées)
      → Rechargés automatiquement au démarrage
      → Sauvegardés à la fermeture ET toutes les 50 étapes
  ✔ Progression visible dans l'UI
      → Compteur de trades gagnants / perdants
      → Win-rate en %
      → Epsilon (taux d'exploration) décroissant avec le temps
"""

import os, threading, time, pickle, warnings
from datetime import datetime
from collections import deque

import numpy as np, pandas as pd, pandas_ta as ta, ccxt
import tkinter as tk
from tkinter import scrolledtext

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION GLOBALE
# ══════════════════════════════════════════════════════════════════════
CONFIG = {
    "symboles"    : ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "timeframe"   : "5m",
    "lookback"    : 250,
    "capital_init": 1000.0,
    "fees"        : 0.001,
    "atrs"        : {
        "sl_mult"      : 1.5,
        "tp_mult"      : 3.0,
        "trail_trigger": 0.015,
        "trail_drop"   : 0.007,
    },
    "dqn"         : {
        "lr"        : 1e-3,
        "gamma"     : 0.95,
        "eps_start" : 1.0,
        "eps_end"   : 0.05,
        "eps_decay" : 0.9995,   # plus lent → exploration plus longue
        "batch"     : 64,
        "mem_size"  : 5000,     # mémoire élargie
        "hidden"    : 64,
    },
    "ui"          : {"max_log_lines": 500},
    "data_sec"    : 30,    # intervalle fetch API (évite le spam)
    "decision_sec": 2,     # intervalle boucle décision (apprentissage temps réel)
}

WEIGHTS_FILE = "dqn_weights.pkl"
MEMORY_FILE  = "dqn_memory.pkl"


# ══════════════════════════════════════════════════════════════════════
#  1. INDICATOR ENGINE — singleton, colonnes par nom
# ══════════════════════════════════════════════════════════════════════
class IndicatorEngine:

    @staticmethod
    def _col(df, prefix):
        cols = [c for c in df.columns if c.startswith(prefix)]
        return cols[0] if cols else None

    @staticmethod
    def _ta(df):
        for length in (9, 21, 50, 200):
            df[f"EMA_{length}"] = ta.ema(df["close"], length=length)

        df["RSI_14"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            col_m = IndicatorEngine._col(macd, "MACD_")
            col_h = IndicatorEngine._col(macd, "MACDh_")
            col_s = IndicatorEngine._col(macd, "MACDs_")
            df["MACD"]        = macd[col_m].values if col_m else 0.0
            df["MACD_HIST"]   = macd[col_h].values if col_h else 0.0
            df["MACD_SIGNAL"] = macd[col_s].values if col_s else 0.0
        else:
            df["MACD"] = df["MACD_HIST"] = df["MACD_SIGNAL"] = 0.0

        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None and not bb.empty:
            col_u = IndicatorEngine._col(bb, "BBU_")
            col_m = IndicatorEngine._col(bb, "BBM_")
            col_l = IndicatorEngine._col(bb, "BBL_")
            if col_u and col_m and col_l:
                df["BB_UPPER"] = bb[col_u].values
                df["BB_MID"]   = bb[col_m].values
                df["BB_LOWER"] = bb[col_l].values
                denom = (df["BB_UPPER"] - df["BB_LOWER"]).replace(0, 1e-8)
                df["BB_POS"] = (df["close"] - df["BB_LOWER"]) / denom
            else:
                df["BB_UPPER"] = df["BB_MID"] = df["BB_LOWER"] = df["close"]
                df["BB_POS"] = 0.5
        else:
            df["BB_UPPER"] = df["BB_MID"] = df["BB_LOWER"] = df["close"]
            df["BB_POS"] = 0.5

        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None and not adx.empty:
            col_adx = IndicatorEngine._col(adx, "ADX_")
            col_dip = IndicatorEngine._col(adx, "DMP_")
            col_dim = IndicatorEngine._col(adx, "DMN_")
            df["ADX"]      = adx[col_adx].values if col_adx else 20.0
            df["DI_PLUS"]  = adx[col_dip].values if col_dip else 20.0
            df["DI_MINUS"] = adx[col_dim].values if col_dim else 20.0
        else:
            df["ADX"] = df["DI_PLUS"] = df["DI_MINUS"] = 20.0

        df["ATR"]  = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["NATR"] = (df["ATR"] / (df["close"] + 1e-8)) * 100

        df["VOL_SMA"]   = df["volume"].rolling(20).mean()
        df["VOL_RATIO"] = df["volume"] / (df["VOL_SMA"] + 1e-8)

        df["OBV"]     = ta.obv(df["close"], df["volume"])
        df["OBV_EMA"] = ta.ema(df["OBV"], length=20)

        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        if stoch is not None and not stoch.empty:
            df["STOCH_K"] = stoch.iloc[:, 0]
            df["STOCH_D"] = stoch.iloc[:, 1]
        else:
            df["STOCH_K"] = df["STOCH_D"] = 50.0

        df["CCI"] = ta.cci(df["high"], df["low"], df["close"], length=20)

        df = df.bfill().ffill().fillna(0)
        return df

    def enrich(self, raw_df):
        try:
            if len(raw_df) < 60:
                return None, None, None

            df     = self._ta(raw_df.copy())
            latest = df.iloc[-1]
            prev   = df.iloc[-2]

            def trend_score():
                s = 50
                if (latest["close"] > latest["EMA_9"] > latest["EMA_21"]
                        > latest["EMA_50"]):
                    s += 25
                elif (latest["close"] < latest["EMA_9"] < latest["EMA_21"]
                        < latest["EMA_50"]):
                    s -= 25
                s += 10 if latest["MACD_HIST"] > 0 else -10
                if latest["ADX"] > 25:
                    s += 10 if latest["DI_PLUS"] > latest["DI_MINUS"] else -10
                return int(max(0, min(100, s)))

            def momentum_score():
                s   = 50
                rsi = float(latest["RSI_14"])
                if   rsi < 30: s += 20
                elif rsi < 50: s += 10
                elif rsi > 70: s -= 20
                elif rsi > 55: s -= 5
                sk_n = float(latest["STOCH_K"]); sk_p = float(prev["STOCH_K"])
                sd_n = float(latest["STOCH_D"]); sd_p = float(prev["STOCH_D"])
                if sk_p < sd_p and sk_n > sd_n:   s += 10
                elif sk_p > sd_p and sk_n < sd_n: s -= 10
                return int(max(0, min(100, s)))

            def vol_score():
                s  = 50
                vr = float(latest["VOL_RATIO"])
                if vr > 1.5:
                    s += 20 if float(latest["close"]) > float(prev["close"]) else -10
                elif vr < 0.7:
                    s -= 15
                s += 10 if float(latest["OBV"]) > float(latest["OBV_EMA"]) else -10
                return int(max(0, min(100, s)))

            def vola_score():
                s    = 50
                natr = float(latest["NATR"])
                if 1.5 <= natr <= 3.5: s += 20
                elif natr > 4.0:       s += 5
                elif natr < 0.5:       s -= 15
                return int(max(0, min(100, s)))

            scores = {
                "tendance"  : trend_score(),
                "momentum"  : momentum_score(),
                "volume"    : vol_score(),
                "volatility": vola_score(),
                "rsi"       : float(latest["RSI_14"]),
                "adx"       : float(latest["ADX"]),
                "bb_pos"    : float(np.clip(latest["BB_POS"], 0, 1)),
                "atr"       : float(latest["ATR"]),
                "close"     : float(latest["close"]),
                "ema200_ok" : int(latest["close"] > latest["EMA_200"]),
            }

            state = np.array([
                scores["tendance"]   / 100.0,
                scores["momentum"]   / 100.0,
                scores["volume"]     / 100.0,
                scores["volatility"] / 100.0,
                min(scores["rsi"] / 100.0, 1.0),
                min(scores["adx"] / 60.0,  1.0),
                scores["bb_pos"],
                min(float(latest["VOL_RATIO"]) / 4.0, 1.0),
                1.0 if float(latest["MACD_HIST"]) > 0 else 0.0,
                float(scores["ema200_ok"]),
            ], dtype=float)

            return df, state, scores

        except Exception as err:
            print(f"[ENRICH ERROR] {err}")
            return None, None, None


# ══════════════════════════════════════════════════════════════════════
#  2. MiniNet — réseau neuronal NumPy
# ══════════════════════════════════════════════════════════════════════
class MiniNet:
    def __init__(self, n_in=10, n_hid=64, n_out=3, lr=1e-3):
        self.lr = lr
        self.W1 = np.random.randn(n_in, n_hid) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hid)
        self.W2 = np.random.randn(n_hid, 32)   * np.sqrt(2.0 / n_hid)
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, n_out)    * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(n_out)

    @staticmethod
    def relu(x):  return np.maximum(0., x)
    @staticmethod
    def drelu(x): return (x > 0).astype(float)

    def forward(self, X):
        self.x  = X
        self.z1 = X @ self.W1 + self.b1;     self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def backward(self, target):
        n   = target.shape[0]
        d3  = (self.z3 - target) / n
        dW3 = self.a2.T @ d3;   db3 = d3.sum(0)
        d2  = (d3 @ self.W3.T) * self.drelu(self.z2)
        dW2 = self.a1.T @ d2;   db2 = d2.sum(0)
        d1  = (d2 @ self.W2.T) * self.drelu(self.z1)
        dW1 = self.x.T  @ d1;   db1 = d1.sum(0)
        clip = 1.0
        for W, dW, b, db in [(self.W3,dW3,self.b3,db3),
                              (self.W2,dW2,self.b2,db2),
                              (self.W1,dW1,self.b1,db1)]:
            W -= self.lr * np.clip(dW, -clip, clip)
            b -= self.lr * np.clip(db, -clip, clip)

    def predict_one(self, state):
        return self.forward(np.array(state, float).reshape(1, -1))[0]


# ══════════════════════════════════════════════════════════════════════
#  3. DQN AGENT — apprentissage temps réel + mémoire complète
# ══════════════════════════════════════════════════════════════════════
class DQNAgent:
    """
    Deep Q-Network temps réel.
    - Apprend à chaque step (pas seulement à la clôture)
    - Sauvegarde poids + replay buffer
    - Recharge tout au démarrage → la mémoire ne se perd jamais
    """

    def __init__(self, config):
        cfg          = config["dqn"]
        self.net     = MiniNet(n_hid=cfg["hidden"], n_out=3, lr=cfg["lr"])
        self.target  = MiniNet(n_hid=cfg["hidden"], n_out=3, lr=0.0)
        self.memory  = deque(maxlen=cfg["mem_size"])
        self.epsilon = cfg["eps_start"]
        self.gamma   = cfg["gamma"]
        self.batch   = cfg["batch"]
        self.steps   = 0
        self.lock    = threading.Lock()
        # Stats progression
        self.wins    = 0
        self.losses  = 0
        self._sync_target()
        self._load_weights()
        self._load_memory()

    def _sync_target(self):
        for a in ("W1","b1","W2","b2","W3","b3"):
            setattr(self.target, a, getattr(self.net, a).copy())

    # ── Persistance poids ─────────────────────────────────────────────
    def _load_weights(self):
        if not os.path.isfile(WEIGHTS_FILE):
            print("[DQN] Aucun poids sauvegardé — démarrage vierge")
            return
        try:
            with open(WEIGHTS_FILE, "rb") as f:
                data = pickle.load(f)
            for a in ("W1","b1","W2","b2","W3","b3"):
                if a in data:
                    setattr(self.net, a, data[a])
            self.epsilon = data.get("eps",   self.epsilon)
            self.steps   = data.get("steps", 0)
            self.wins    = data.get("wins",  0)
            self.losses  = data.get("losses",0)
            self._sync_target()
            print(f"[DQN] ✅ Poids chargés — ε={self.epsilon:.4f}  "
                  f"steps={self.steps}  W/L={self.wins}/{self.losses}")
        except Exception as e:
            print(f"[DQN] Erreur chargement poids : {e}")

    def _save_weights(self):
        try:
            data = {a: getattr(self.net, a)
                    for a in ("W1","b1","W2","b2","W3","b3")}
            data["eps"]    = self.epsilon
            data["steps"]  = self.steps
            data["wins"]   = self.wins
            data["losses"] = self.losses
            with open(WEIGHTS_FILE, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[DQN] Erreur sauvegarde poids : {e}")

    # ── Persistance mémoire (replay buffer) ───────────────────────────
    def _load_memory(self):
        if not os.path.isfile(MEMORY_FILE):
            print("[DQN] Aucune mémoire sauvegardée — replay buffer vide")
            return
        try:
            with open(MEMORY_FILE, "rb") as f:
                saved = pickle.load(f)
            self.memory = deque(saved, maxlen=self.memory.maxlen)
            print(f"[DQN] ✅ Mémoire chargée — {len(self.memory)} transitions")
        except Exception as e:
            print(f"[DQN] Erreur chargement mémoire : {e}")

    def _save_memory(self):
        try:
            with open(MEMORY_FILE, "wb") as f:
                pickle.dump(list(self.memory), f)
        except Exception as e:
            print(f"[DQN] Erreur sauvegarde mémoire : {e}")

    def save_all(self):
        """Sauvegarde complète : poids + mémoire."""
        self._save_weights()
        self._save_memory()
        print(f"[DQN] 💾 Tout sauvegardé — {len(self.memory)} transitions, "
              f"steps={self.steps}, W/L={self.wins}/{self.losses}")

    # ── Décision ε-greedy ─────────────────────────────────────────────
    def choose_action(self, state):
        with self.lock:
            eps = self.epsilon
        if np.random.rand() < eps:
            return int(np.random.choice([0, 1, 2]))
        return int(np.argmax(self.net.predict_one(state)))

    # ── Mémorisation ──────────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        with self.lock:
            self.memory.append((
                np.array(state,      dtype=float),
                int(action),
                float(reward),
                np.array(next_state, dtype=float),
                bool(done),
            ))

    # ── Apprentissage Bellman (temps réel) ────────────────────────────
    def learn(self):
        with self.lock:
            n = len(self.memory)
        if n < self.batch:
            return

        with self.lock:
            idx   = np.random.choice(n, self.batch, replace=False)
            batch = [self.memory[i] for i in idx]

        states      = np.array([b[0] for b in batch], dtype=float)
        actions     = np.array([b[1] for b in batch], dtype=int)
        rewards     = np.array([b[2] for b in batch], dtype=float)
        next_states = np.array([b[3] for b in batch], dtype=float)
        dones       = np.array([b[4] for b in batch], dtype=bool)

        q_next_max = self.target.forward(next_states).max(axis=1)
        targets    = self.net.forward(states).copy()

        for i in range(self.batch):
            targets[i, actions[i]] = (
                rewards[i] if dones[i]
                else rewards[i] + self.gamma * q_next_max[i]
            )

        with self.lock:
            self.net.backward(targets)
            self.steps += 1
            if self.epsilon > CONFIG["dqn"]["eps_end"]:
                self.epsilon *= CONFIG["dqn"]["eps_decay"]
            if self.steps % 100 == 0:
                self._sync_target()
            # Sauvegarde périodique (toutes les 50 étapes d'apprentissage)
            if self.steps % 50 == 0:
                self._save_weights()

    def record_trade(self, pnl):
        """Compte les trades gagnants/perdants pour la progression."""
        with self.lock:
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1


# ══════════════════════════════════════════════════════════════════════
#  4. EXCHANGE MANAGER
# ══════════════════════════════════════════════════════════════════════
class ExchangeManager:
    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

    def fetch_ohlcv(self, symbol, timeframe, limit):
        try:
            return self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit
            )
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"[EXCH] {symbol} – {type(e).__name__}: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════
#  5. SYMBOL TRADER — apprentissage temps réel (boucle 2s)
# ══════════════════════════════════════════════════════════════════════
class SymbolTrader(threading.Thread):
    """
    Architecture double-thread :
    ┌─ _data_loop()    : fetch OHLCV + indicateurs toutes les 30s
    └─ run()           : décision DQN + apprentissage toutes les 2s

    Apprentissage continu :
    - Chaque step = 1 transition stockée + learn() appelé
    - Récompense intermédiaire = variation de prix depuis la step précédente
    - Récompense terminale = PnL% complet - pénalité durée
    """

    def __init__(self, symbol, exchange, agent, capital, cfg,
                 ui_callback, log_callback):
        super().__init__(daemon=True, name=f"Trader-{symbol}")
        self.symbol      = symbol
        self.exchange    = exchange
        self.agent       = agent
        self.capital     = capital
        self.capital_ini = capital
        self.cfg         = cfg
        self.ui_cb       = ui_callback
        self.log_cb      = log_callback

        # État de la position
        self.pos_active       = False
        self.entry_price      = 0.0
        self.max_price        = 0.0
        self.stop_loss        = 0.0
        self.tp1              = 0.0
        self.tp2              = 0.0
        self._trail_triggered = False
        self._tp1_hit         = False
        self._state_entry     = None
        self._step_entry      = 0

        # Données partagées entre data_loop et run
        self._data_lock     = threading.Lock()
        self._latest_state  = None
        self._latest_scores = None
        self._latest_price  = 0.0
        self._latest_atr    = 0.0
        self._prev_price    = 0.0   # pour la récompense intermédiaire

        self._lock = threading.Lock()
        self._run  = True
        self._ie   = IndicatorEngine()

    # ── Fetch données (30s) ──────────────────────────────────────────
    def _get_data(self):
        raw = self.exchange.fetch_ohlcv(
            self.symbol,
            timeframe=self.cfg["timeframe"],
            limit=self.cfg["lookback"],
        )
        if raw is None:
            return None
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df.astype({"open":float,"high":float,"low":float,
                          "close":float,"volume":float})

    def _data_loop(self):
        """Thread séparé : rafraîchit les données toutes les 30s."""
        while self._run:
            try:
                df = self._get_data()
                if df is not None:
                    df_e, state, scores = self._ie.enrich(df)
                    if df_e is not None:
                        price = float(df_e["close"].iloc[-1])
                        atr   = float(df_e["ATR"].iloc[-1])
                        with self._data_lock:
                            self._latest_state  = state
                            self._latest_scores = scores
                            self._latest_price  = price
                            self._latest_atr    = atr
            except Exception as e:
                self.log_cb(self.symbol, f"⚠️ Data: {e}")
            time.sleep(self.cfg["data_sec"])

    # ── Signal technique 5/7 ─────────────────────────────────────────
    def _signal_technique(self, scores):
        cond = 0
        if scores["tendance"]   > 65:   cond += 1
        if scores["momentum"]   > 60:   cond += 1
        if scores["volume"]     > 60:   cond += 1
        if scores["volatility"] > 50:   cond += 1
        if 38 < scores["rsi"] < 65:     cond += 1
        if scores["adx"]        > 20:   cond += 1
        if scores["bb_pos"]     < 0.90: cond += 1
        if cond >= 5: return 1
        if cond <= 2 and self.pos_active: return 2
        return 0

    # ── Ouverture de position ────────────────────────────────────────
    def _enter_position(self, price, atr, state):
        with self._lock:
            self.pos_active       = True
            self.entry_price      = price
            self.max_price        = price
            self._trail_triggered = False
            self._tp1_hit         = False
            self._state_entry     = state.copy()
            self._step_entry      = self.agent.steps
            self.stop_loss = price - atr * self.cfg["atrs"]["sl_mult"]
            self.tp1       = price + atr * self.cfg["atrs"]["tp_mult"] * 0.5
            self.tp2       = price + atr * self.cfg["atrs"]["tp_mult"]
            self.capital  *= (1 - self.cfg["fees"])
        self.log_cb(self.symbol,
                    f"🟢 BUY @ {price:,.5f}  ┊  "
                    f"SL {self.stop_loss:,.5f}  ┊  "
                    f"TP {self.tp2:,.5f}  ┊  ATR {atr:.5f}")

    # ── Clôture de position ──────────────────────────────────────────
    def _exit_position(self, price, reason, current_state):
        with self._lock:
            pnl  = (price - self.entry_price) / self.entry_price
            self.capital         *= (1 + pnl) * (1 - self.cfg["fees"])
            self.pos_active       = False
            self._trail_triggered = False
            self._tp1_hit         = False
            state_entry           = (self._state_entry.copy()
                                     if self._state_entry is not None
                                     else current_state)
            durée = max(self.agent.steps - self._step_entry, 1)
            cap   = self.capital

        # Récompense terminale = PnL% - pénalité durée
        penalite = 0.01 * durée
        reward   = pnl * 100.0 - penalite

        # Stocker transition terminale et apprendre
        self.agent.store(state_entry, 1, reward, current_state, True)
        self.agent.learn()
        self.agent.record_trade(pnl)

        icone = "💚" if pnl > 0 else "🔴"
        self.log_cb(self.symbol,
                    f"{icone} CLOSE {reason} @ {price:,.5f}  ┊  "
                    f"PnL {pnl*100:+.2f}%  ┊  Capital ${cap:,.2f}")
        self.ui_cb(self.symbol, price, pnl * 100.0, cap)
        return pnl

    # ── Surveillance position ────────────────────────────────────────
    def _monitor_position(self, price, current_state):
        if price <= self.stop_loss:
            self._exit_position(price, "STOP-LOSS", current_state)
            return True
        if price >= self.tp2:
            self._exit_position(price, "TAKE-PROFIT", current_state)
            return True
        if price >= self.tp1 and not self._tp1_hit:
            self._tp1_hit = True
            with self._lock:
                self.stop_loss = self.entry_price
            self.log_cb(self.symbol,
                        f"🔒 TP1 → breakeven @ {self.entry_price:,.5f}")
        with self._lock:
            if price > self.max_price:
                self.max_price = price
            pnl   = (price - self.entry_price) / self.entry_price
            chute = ((self.max_price - price) / self.max_price
                     if self.max_price > 0 else 0.0)
        trail_trigger = self.cfg["atrs"]["trail_trigger"]
        trail_drop    = self.cfg["atrs"]["trail_drop"]
        if pnl >= trail_trigger and chute >= trail_drop:
            if not self._trail_triggered:
                self._trail_triggered = True
                self._exit_position(price, "TRAILING-STOP", current_state)
                return True
        return False

    # ── Boucle principale (2s) — APPRENTISSAGE TEMPS RÉEL ────────────
    def run(self):
        self.log_cb(self.symbol, f"▶  Thread démarré — capital ${self.capital:,.2f}")

        threading.Thread(target=self._data_loop, daemon=True, name=f"Data-{self.symbol}").start()

        while self._run:
            try:
                with self._data_lock:
                    state  = self._latest_state
                    scores = self._latest_scores
                    price  = self._latest_price
                    atr    = self._latest_atr

                if state is None:
                    time.sleep(2)
                    continue

                # --- AJOUT : STOP LOSS DE SÉCURITÉ ABSOLUE (1%) ---
                if self.pos_active:
                    pnl_actuel = (price - self.entry_price) / self.entry_price
                    if pnl_actuel <= -0.01: # Si perte > 1%, on coupe tout
                        self._exit_position(price, "SÉCURITÉ-1%", state)
                        continue

                # ── PnL latent temps réel ────────────────────────────
                pnl_latent = (price - self.entry_price) / self.entry_price * 100.0 if self.pos_active else 0.0
                self.ui_cb(self.symbol, price, pnl_latent, self.capital)

                # ── Récompense intermédiaire ─────
                step_reward = (price - self._prev_price) / self._prev_price * 50.0 if (self._prev_price > 0 and self.pos_active) else 0.0
                self._prev_price = price

                # ── Décision DQN + signal technique ─────────────────
                action_dqn = self.agent.choose_action(state)
                sig_tech   = self._signal_technique(scores)

                # ── GESTION POSITION AVEC FILTRE DE FRAIS ──────────
                if self.pos_active:
                    exited = self._monitor_position(price, state)
                    if exited:
                        self._prev_price = 0.0
                        time.sleep(self.cfg["decision_sec"])
                        continue
                    
                    # --- AJOUT : FILTRE DE SORTIE (Ne pas vendre si on perd juste à cause des frais) ---
                    if action_dqn == 2:
                        frais_estimés = self.cfg["fees"] * 2 # Achat + Vente
                        if pnl_latent/100 > frais_estimés or pnl_latent/100 < -0.005:
                            self._exit_position(price, "DQN-SELL", state)
                            self._prev_price = 0.0
                        else:
                            # L'IA veut vendre mais on perdrait trop en frais, on attend un meilleur prix
                            self.agent.store(state, 0, step_reward, state, False)
                            self.agent.learn()
                    else:
                        self.agent.store(state, 0, step_reward, state, False)
                        self.agent.learn()
                else:
                    # --- AJOUT : FILTRE D'ENTRÉE (Confiance IA + Signal Tech) ---
                    # On n'entre que si l'IA dit BUY (1) ET que les indicateurs sont OK
                    if action_dqn == 1 and sig_tech == 1:
                        # On vérifie que la volatilité (ATR) n'est pas trop faible pour couvrir les frais
                        if (atr / price) > (self.cfg["fees"] * 3):
                            self._enter_position(price, atr, state)
                        else:
                            self.log_cb(self.symbol, "⏳ Signal ignoré : Volatilité trop faible pour les frais")
                    else:
                        self.agent.store(state, action_dqn, 0.0, state, False)
                        self.agent.learn()

            except Exception as e:
                self.log_cb(self.symbol, f"⚠️  Erreur : {e}")
                time.sleep(5)

            time.sleep(self.cfg["decision_sec"])

    def stop(self):
        self._run = False


# ══════════════════════════════════════════════════════════════════════
#  6. UI TKINTER — thread-safe, stats progression
# ══════════════════════════════════════════════════════════════════════
class BotUI:
    def __init__(self, root, symbols, cfg, exchange, agent, capital_per_sym):
        self.root        = root
        self.cfg         = cfg
        self.symbols     = symbols
        self.exchange    = exchange
        self.agent       = agent
        self.capital_ini = capital_per_sym
        self.capital     = {s: capital_per_sym for s in symbols}

        self.root.title("BOT TRADING COMPOSITE v2 — APPRENTISSAGE TEMPS RÉEL")
        self.root.configure(bg="#0d0d0d")
        self.root.geometry("900x720")

        self._build_ui(symbols, capital_per_sym)

    def _build_ui(self, symbols, capital_per_sym):
        tk.Label(
            self.root,
            text="🤖  BOT TRADING v2  ·  DQN TEMPS RÉEL  ·  BTC · ETH · SOL",
            fg="#00ffcc", bg="#0d0d0d", font=("Arial", 13, "bold")
        ).pack(pady=8)

        # Tableau symboles
        tbl = tk.Frame(self.root, bg="#111111", relief=tk.RIDGE, bd=1)
        tbl.pack(fill=tk.X, padx=12, pady=4)

        self.lbl_price = {}
        self.lbl_pnl   = {}
        self.lbl_cap   = {}

        for col, hdr in enumerate(["Symbole","Prix ($)","PnL latent","Capital ($)"]):
            tk.Label(tbl, text=hdr, fg="#888888", bg="#111111",
                     font=("Consolas",10,"bold"), width=20,
                     anchor="center").grid(row=0, column=col, padx=3, pady=4)

        for r, sym in enumerate(symbols, start=1):
            tk.Label(tbl, text=sym, fg="#ffd700", bg="#111111",
                     font=("Consolas",10,"bold"), width=20,
                     anchor="center").grid(row=r, column=0, padx=3, pady=3)
            self.lbl_price[sym] = tk.Label(tbl, text="…", fg="white",
                bg="#111111", font=("Consolas",9), width=20, anchor="center")
            self.lbl_price[sym].grid(row=r, column=1, padx=3)
            self.lbl_pnl[sym] = tk.Label(tbl, text="0.00 %", fg="white",
                bg="#111111", font=("Consolas",9), width=20, anchor="center")
            self.lbl_pnl[sym].grid(row=r, column=2, padx=3)
            self.lbl_cap[sym] = tk.Label(
                tbl, text=f"{capital_per_sym:,.2f}", fg="#00ff66",
                bg="#111111", font=("Consolas",9), width=20, anchor="center")
            self.lbl_cap[sym].grid(row=r, column=3, padx=3)

        # Barre progression DQN
        frm_dqn = tk.Frame(self.root, bg="#0d0d0d")
        frm_dqn.pack(fill=tk.X, padx=12, pady=4)

        self.lbl_dqn = tk.Label(
            frm_dqn, text="[DQN] Initialisation…",
            fg="#00aaff", bg="#0d0d0d", font=("Consolas",9), anchor="w")
        self.lbl_dqn.pack(side=tk.LEFT)

        self.lbl_winrate = tk.Label(
            frm_dqn, text="Win-rate : —",
            fg="#ffaa00", bg="#0d0d0d", font=("Consolas",9,"bold"), anchor="e")
        self.lbl_winrate.pack(side=tk.RIGHT, padx=10)

        # Journal
        tk.Label(self.root, text="📋  Journal des événements",
                 fg="#888888", bg="#0d0d0d",
                 font=("Arial",10)).pack(anchor=tk.W, padx=15, pady=(6,0))

        fr = tk.Frame(self.root, bg="#0d0d0d")
        fr.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self.log_txt = scrolledtext.ScrolledText(
            fr, height=22, bg="#000000", fg="#00ff00",
            font=("Consolas",9), state=tk.DISABLED, wrap=tk.WORD)
        self.log_txt.pack(fill=tk.BOTH, expand=True)

    # ── Callbacks thread-safe ────────────────────────────────────────
    def _ui_cb(self, sym, prix, pnl, cap):
        def _upd():
            self.lbl_price[sym].config(text=f"{prix:,.4f}")
            fg_p = "#00ff66" if pnl >= 0 else "#ff4444"
            self.lbl_pnl[sym].config(text=f"{pnl:+.2f} %", fg=fg_p)
            fg_c = "#00ff66" if cap >= self.capital_ini else "#ff4444"
            self.lbl_cap[sym].config(text=f"{cap:,.2f}", fg=fg_c)
        self.root.after(0, _upd)

    def _log_cb(self, sym, msg):
        ts  = datetime.now().strftime("%H:%M:%S")
        txt = f"{ts} [{sym:<9}]  {msg}\n"
        def _up():
            self.log_txt.configure(state=tk.NORMAL)
            self.log_txt.insert(tk.END, txt)
            total = int(self.log_txt.index("end-1c").split(".")[0])
            if total > self.cfg["ui"]["max_log_lines"]:
                self.log_txt.delete("1.0",
                    f"{total - self.cfg['ui']['max_log_lines']}.0")
            self.log_txt.see(tk.END)
            self.log_txt.configure(state=tk.DISABLED)
        self.root.after(0, _up)

    # ── Démarrage ────────────────────────────────────────────────────
    def start_traders(self):
        self._traders = []
        for sym in self.symbols:
            t = SymbolTrader(
                symbol       = sym,
                exchange     = self.exchange,
                agent        = self.agent,
                capital      = self.capital[sym],
                cfg          = self.cfg,
                ui_callback  = self._ui_cb,
                log_callback = self._log_cb,
            )
            t.start()
            self._traders.append(t)

        # Thread moniteur : mise à jour UI DQN
        def _monitor():
            while True:
                eps   = self.agent.epsilon
                mem   = len(self.agent.memory)
                steps = self.agent.steps
                w     = self.agent.wins
                l     = self.agent.losses
                total = w + l
                wr    = f"{w/total*100:.1f}%" if total > 0 else "—"

                txt_dqn = (f"[DQN]  ε={eps:.4f}  ┊  "
                           f"Replay {mem}/{self.cfg['dqn']['mem_size']}  ┊  "
                           f"Steps {steps}  ┊  "
                           f"Trades {total}")
                txt_wr  = f"Win-rate : {wr}  ({w}W / {l}L)"

                self.root.after(0, lambda t=txt_dqn: self.lbl_dqn.config(text=t))
                self.root.after(0, lambda t=txt_wr:  self.lbl_winrate.config(text=t))
                self.root.after(0, lambda t=f"BOT v2 — ε={eps:.4f} | {wr}":
                                self.root.title(t))
                time.sleep(2)

        threading.Thread(target=_monitor, daemon=True).start()

    def on_close(self):
        self._log_cb("SYSTÈME", "🛑 Arrêt en cours — sauvegarde de la mémoire…")
        for t in self._traders:
            t.stop()
        # Sauvegarder poids ET replay buffer
        self.agent.save_all()
        self.root.after(500, self.root.destroy)


# ══════════════════════════════════════════════════════════════════════
#  7. MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  BOT TRADING COMPOSITE v2 — APPRENTISSAGE TEMPS RÉEL")
    print("  BTC/USDT · ETH/USDT · SOL/USDT  |  DQN 2s  |  Data 30s")
    print(f"  Poids : {WEIGHTS_FILE}  |  Mémoire : {MEMORY_FILE}")
    print("=" * 65)

    root = tk.Tk()
    ui   = BotUI(
        root,
        symbols         = CONFIG["symboles"],
        cfg             = CONFIG,
        exchange        = ExchangeManager(),
        agent           = DQNAgent(CONFIG),
        capital_per_sym = CONFIG["capital_init"],
    )
    ui.start_traders()
    root.protocol("WM_DELETE_WINDOW", ui.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
