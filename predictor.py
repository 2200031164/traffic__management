# predictor.py
"""
Robust traffic predictor with LSTM (if TensorFlow available),
MLP (if scikit-learn available), and moving-average fallback.

API:
    init_predictor(junction_names)
    update_and_predict(current_totals) -> returns dict {junction: predicted_value}
    predict_next_totals() -> returns dict (prediction without adding a new observation)
"""

import logging
import time
from collections import deque

import numpy as np

# configure lightweight logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Try TensorFlow LSTM
TF_AVAILABLE = False
try:
    import tensorflow as tf  # optional
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
    logger.info("TensorFlow available: using LSTM when requested.")
except Exception:
    TF_AVAILABLE = False

# Try scikit-learn MLPRegressor (fallback)
SKL_AVAILABLE = False
try:
    from sklearn.neural_network import MLPRegressor
    SKL_AVAILABLE = True
    logger.info("scikit-learn available: can use MLPRegressor when requested.")
except Exception:
    SKL_AVAILABLE = False


class TrafficPredictor:
    def __init__(self, junction_names, history_len=20, window=5, horizon=1,
                 train_every=10, model_type='auto'):
        """
        junction_names: iterable of junction ids
        history_len: how many past counts to keep
        window: input length used for model training/prediction
        horizon: prediction horizon (we support horizon==1 best)
        train_every: retrain every N frames (non-zero integer)
        model_type: 'auto', 'lstm', 'mlp', or 'ma' (moving average)
        """
        self.junctions = list(junction_names)
        self.history_len = int(history_len)
        self.window = int(window)
        self.horizon = int(horizon)
        self.train_every = max(1, int(train_every))
        self.frame_counter = 0

        # history per junction
        self.history = {j: deque(maxlen=self.history_len) for j in self.junctions}

        # store models per junction
        self.models = {j: None for j in self.junctions}

        # choose model type
        if model_type == 'auto':
            if TF_AVAILABLE:
                self.model_type = 'lstm'
            elif SKL_AVAILABLE:
                self.model_type = 'mlp'
            else:
                self.model_type = 'ma'
        else:
            self.model_type = model_type

        # small training params
        self.lstm_epochs = 6
        self.mlp_max_iter = 300

    def add_observation(self, obs_dict):
        """obs_dict: {junction: integer_count}"""
        for j, val in obs_dict.items():
            if j in self.history:
                try:
                    self.history[j].append(float(val))
                except Exception:
                    logger.debug("Could not append val for %s: %r", j, val)

    def _make_training_data(self, arr):
        """arr: 1D list-like. returns (X, y) or (None, None)"""
        arr = list(arr)
        n = len(arr)
        if n < self.window + self.horizon:
            return None, None
        X = []
        y = []
        for i in range(n - self.window - self.horizon + 1):
            X.append(arr[i:i + self.window])
            y.append(arr[i + self.window:i + self.window + self.horizon])
        return np.array(X, dtype=float), np.array(y, dtype=float)

    def _build_lstm_model(self, input_shape, horizon):
        """Build a small LSTM model (input_shape = (timesteps, features))"""
        model = Sequential()
        model.add(LSTM(24, activation='tanh', input_shape=input_shape))
        model.add(Dense(horizon))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_if_needed(self):
        """Called each frame; actually trains only every self.train_every frames."""
        self.frame_counter += 1
        if (self.frame_counter % self.train_every) != 0:
            return

        for j in self.junctions:
            data = list(self.history[j])
            X, y = self._make_training_data(data)
            if X is None:
                continue

            try:
                if self.model_type == 'lstm' and TF_AVAILABLE:
                    # reshape X -> (samples, timesteps, 1)
                    Xr = X.reshape((X.shape[0], X.shape[1], 1))
                    model = self.models.get(j)
                    if model is None:
                        model = self._build_lstm_model((self.window, 1), self.horizon)
                        self.models[j] = model
                    # train briefly
                    model.fit(Xr, y, epochs=self.lstm_epochs, verbose=0, batch_size=8)
                elif self.model_type == 'mlp' and SKL_AVAILABLE:
                    model = self.models.get(j)
                    # MLPRegressor doesn't support partial_fit for regression reliably here;
                    # we simply fit from scratch for small datasets.
                    mlp = MLPRegressor(hidden_layer_sizes=(32, ), max_iter=self.mlp_max_iter)
                    try:
                        mlp.fit(X, y.ravel())
                        self.models[j] = mlp
                    except Exception as e:
                        logger.debug("MLP training failed for %s: %s", j, e)
                else:
                    # moving-average needs no training
                    pass
            except Exception as e:
                logger.debug("Training error for junction %s: %s", j, e)

    def predict_next(self):
        """Return dict {junction: predicted_value} for the next step (horizon=1 expected)."""
        preds = {}
        for j in self.junctions:
            data = list(self.history[j])
            if len(data) == 0:
                preds[j] = 0.0
                continue

            model = self.models.get(j)

            try:
                if self.model_type == 'lstm' and model is not None and TF_AVAILABLE and len(data) >= self.window:
                    x = np.array(data[-self.window:], dtype=float).reshape((1, self.window, 1))
                    p = model.predict(x, verbose=0)[0]
                    preds[j] = float(p[0]) if p.size > 0 else float(np.mean(data[-self.window:]))
                elif self.model_type == 'mlp' and model is not None and SKL_AVAILABLE and len(data) >= self.window:
                    x = np.array(data[-self.window:], dtype=float).reshape((1, -1))
                    try:
                        p = model.predict(x)[0]
                        preds[j] = float(p)
                    except Exception:
                        preds[j] = float(np.mean(data[-min(len(data), self.window):]))
                else:
                    # moving-average fallback (use last-window mean)
                    window_vals = data[-min(len(data), self.window):]
                    preds[j] = float(np.mean(window_vals))
            except Exception as e:
                logger.debug("Predict error for %s: %s", j, e)
                preds[j] = float(np.mean(data[-min(len(data), self.window):]))

        return preds


# ===== Convenience global predictor instance and helper functions =====
_predictor = None


def init_predictor(junction_names, **kwargs):
    """Create a global TrafficPredictor instance."""
    global _predictor
    _predictor = TrafficPredictor(junction_names, **kwargs)
    return _predictor


def update_and_predict(current_totals):
    """
    Add current_totals observation and return predicted next totals.
    current_totals: dict {junction_name: integer_count}
    returns: dict {junction_name: predicted_float}
    """
    global _predictor
    if _predictor is None:
        # lazily initialize with keys from the passed dict
        init_predictor(list(current_totals.keys()))
    _predictor.add_observation(current_totals)
    _predictor.train_if_needed()
    return _predictor.predict_next()


def predict_next_totals():
    """Return prediction without updating history first."""
    global _predictor
    if _predictor is None:
        return {}
    return _predictor.predict_next()


# Optional simple smoke test when run directly
if __name__ == "__main__":
    # quick sanity check
    init_predictor(["A", "B", "C"])
    for t in range(15):
        obs = {"A": np.random.poisson(3 + 0.2 * t), "B": np.random.poisson(5), "C": np.random.poisson(1)}
        preds = update_and_predict(obs)
        print(f"t={t} obs={obs} preds={preds}")
