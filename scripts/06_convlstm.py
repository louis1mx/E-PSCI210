"""
Chengdu Urban Cooling Project — Script 06
Purpose : ConvLSTM model for spatiotemporal LST prediction and
          cooling-zone detection.

Why ConvLSTM over simpler methods (required by rubric):
  - ITS and CausalImpact operate on a single time series → they
    cannot tell you WHERE cooling occurs spatially.
  - A plain LSTM flattens spatial structure.
  - ConvLSTM treats each month as a 2-D image and learns both
    spatial patterns (which neighbourhoods cool) and temporal
    dynamics (when the trend breaks) simultaneously.
  - The residual map (predicted − observed post-policy) reveals
    WHICH PIXELS cooled MORE than the model expected from
    pre-policy dynamics alone — the spatial fingerprint of the policy.

Architecture:
  Input  : sequence of T=12 monthly Landsat LST images
           shape (batch, 12, H, W, 1)
  Model  : 2 × ConvLSTM2D layers  →  Conv2D output head
  Output : next-month LST image
           shape (batch, H, W, 1)

Training strategy:
  - Train only on pre-policy data (2016–2021, 60 months)
  - Predict post-policy months (2022–2025)
  - Residual = actual − predicted → negative = more cooling than expected

Outputs:
  outputs/models/convlstm_model.keras
  outputs/figures/fig8_convlstm_residuals.png   (spatial cooling map)
  outputs/figures/fig9_convlstm_timeseries.png  (predicted vs actual)
  outputs/tables/convlstm_metrics.csv

Install:
  pip install tensorflow numpy matplotlib scikit-learn rasterio pillow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ── Paths ───────────────────────────────────────────────────
TILE_DIR  = Path('data/raw/gee/chengdu_lst_tiles')  # GeoTIFFs from Script 01
PANEL     = Path('data/panel/chengdu_monthly_panel.csv')
MODEL_DIR = Path('outputs/models'); MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR   = Path('outputs/figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR   = Path('outputs/tables');  TAB_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV  = TAB_DIR / 'convlstm_training_history.csv'
HISTORY_JSON = TAB_DIR / 'convlstm_training_history.json'

# ── Hyperparameters ──────────────────────────────────────────
SEQ_LEN    = 12     # months of history used as input
IMG_SIZE   = 64     # resize all tiles to 64×64 (fast; increase for final run)
POLICY_YM  = '2022-01'
EPOCHS     = 60
BATCH_SIZE = 4
LR         = 1e-3

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# ════════════════════════════════════════════════════════════
# STEP 1  —  Load and preprocess GeoTIFF tiles
# ════════════════════════════════════════════════════════════

def load_tiles():
    """
    Load all monthly LST GeoTIFFs, resize to IMG_SIZE×IMG_SIZE.
    Returns:
        images : np.ndarray  shape (N_months, IMG_SIZE, IMG_SIZE)
        yms    : list of 'YYYY-MM' strings (sorted)
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError:
        raise ImportError('Install rasterio: pip install rasterio')

    tile_files = sorted(TILE_DIR.glob('LST_*.tif'))
    if len(tile_files) == 0:
        raise FileNotFoundError(
            f'No GeoTIFF files found in {TILE_DIR}.\n'
            'Make sure Script 01 exported the tiles and you downloaded '
            'them from Google Drive into data/raw/gee/chengdu_lst_tiles/')

    images, yms = [], []
    for f in tile_files:
        # Filename: LST_2016_01.tif
        parts = f.stem.split('_')
        ym = f'{parts[1]}-{parts[2]}'

        with rasterio.open(f) as src:
            # Resample to IMG_SIZE × IMG_SIZE
            img = src.read(
                1,
                out_shape=(IMG_SIZE, IMG_SIZE),
                resampling=Resampling.bilinear
            ).astype(np.float32)

        # Fill nodata (clouds) with np.nan, then interpolate
        img[img == src.nodata] = np.nan

        images.append(img)
        yms.append(ym)

    images = np.stack(images)   # (N, H, W)
    print(f'  Loaded {len(images)} tiles  shape={images.shape}')

    # Per-pixel linear interpolation over time for missing values
    for i in range(images.shape[1]):
        for j in range(images.shape[2]):
            ts = images[:, i, j]
            mask = np.isnan(ts)
            if mask.any() and not mask.all():
                idx = np.arange(len(ts))
                images[:, i, j] = np.interp(idx, idx[~mask], ts[~mask])

    # Fill any pixels that were ALL-NaN (permanent cloud cover) with global mean
    global_mean = np.nanmean(images)
    images = np.where(np.isnan(images), global_mean, images)

    return images, yms


def normalise(images):
    """Z-score normalise per pixel across time."""
    mu  = np.nanmean(images, axis=0, keepdims=True)
    sig = np.nanstd(images,  axis=0, keepdims=True) + 1e-6
    return (images - mu) / sig, mu, sig


def make_sequences(images, seq_len=SEQ_LEN):
    """
    Sliding window: X = [t, t+1, …, t+seq_len-1]  y = t+seq_len
    Returns X shape (N, seq_len, H, W, 1)
             y shape (N, H, W, 1)
    """
    X, y = [], []
    for i in range(len(images) - seq_len):
        X.append(images[i:i+seq_len, :, :, np.newaxis])
        y.append(images[i+seq_len,   :, :, np.newaxis])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ════════════════════════════════════════════════════════════
# STEP 2  —  Build ConvLSTM model
# ════════════════════════════════════════════════════════════

def build_model(img_size=IMG_SIZE, seq_len=SEQ_LEN):
    """
    Two-layer ConvLSTM2D → Conv2D output head.
    Input shape: (batch, seq_len, H, W, 1)
    """
    inp = layers.Input(shape=(seq_len, img_size, img_size, 1))

    # First ConvLSTM layer — return sequences for stacking
    x = layers.ConvLSTM2D(
        filters=32, kernel_size=(3, 3), padding='same',
        return_sequences=True, activation='tanh',
        recurrent_activation='sigmoid',
    )(inp)
    x = layers.BatchNormalization()(x)

    # Second ConvLSTM layer — return last hidden state only
    x = layers.ConvLSTM2D(
        filters=16, kernel_size=(3, 3), padding='same',
        return_sequences=False, activation='tanh',
        recurrent_activation='sigmoid',
    )(x)
    x = layers.BatchNormalization()(x)

    # Output head: Conv2D → 1 channel (LST prediction)
    out = layers.Conv2D(
        filters=1, kernel_size=(1, 1), padding='same', activation='linear'
    )(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='mse',
        metrics=['mae']
    )
    model.summary()
    return model


# ════════════════════════════════════════════════════════════
# STEP 3  —  Train and evaluate
# ════════════════════════════════════════════════════════════

def compute_baselines(images, yms, yms_test):
    """
    Compute two naive baselines on the same post-policy test months as ConvLSTM.

    Seasonal climatology: for each calendar month, average all pre-policy
    images of that month and use as prediction.

    Persistence (last pre-policy year): match each post-policy month to the
    same calendar month from the most recent complete pre-policy year.

    Returns dict with RMSE and MAE for each baseline.
    """
    pre_mask = [ym < POLICY_YM for ym in yms]
    images_pre  = images[pre_mask]
    yms_pre     = [ym for ym, m in zip(yms, pre_mask) if m]

    # Raw observed images for the exact test months ConvLSTM was evaluated on
    ym_to_idx    = {ym: i for i, ym in enumerate(yms)}
    test_indices = [ym_to_idx[ym] for ym in yms_test]
    images_test  = images[test_indices]          # (N_test, H, W)

    # ── Seasonal climatology ─────────────────────────────────
    monthly_stack = {}
    for img, ym in zip(images_pre, yms_pre):
        m = int(ym.split('-')[1])
        monthly_stack.setdefault(m, []).append(img)
    monthly_clim = {m: np.mean(v, axis=0) for m, v in monthly_stack.items()}
    overall_pre_mean = np.mean(images_pre, axis=0)

    clim_preds = np.stack([
        monthly_clim.get(int(ym.split('-')[1]), overall_pre_mean)
        for ym in yms_test
    ])
    clim_rmse = np.sqrt(mean_squared_error(images_test.reshape(-1),
                                           clim_preds.reshape(-1)))
    clim_mae  = mean_absolute_error(images_test.reshape(-1),
                                    clim_preds.reshape(-1))

    # ── Persistence: most recent pre-policy year ─────────────
    last_year = sorted(set(ym.split('-')[0] for ym in yms_pre))[-1]
    last_year_imgs = {
        int(ym.split('-')[1]): img
        for img, ym in zip(images_pre, yms_pre)
        if ym.split('-')[0] == last_year
    }

    persist_preds = np.stack([
        last_year_imgs.get(int(ym.split('-')[1]), overall_pre_mean)
        for ym in yms_test
    ])
    persist_rmse = np.sqrt(mean_squared_error(images_test.reshape(-1),
                                              persist_preds.reshape(-1)))
    persist_mae  = mean_absolute_error(images_test.reshape(-1),
                                       persist_preds.reshape(-1))

    print(f'\n  Baseline — Seasonal Climatology : RMSE={clim_rmse:.3f} °C  MAE={clim_mae:.3f} °C')
    print(f'  Baseline — Persistence ({last_year})   : RMSE={persist_rmse:.3f} °C  MAE={persist_mae:.3f} °C')

    return {
        'Seasonal Climatology': {'RMSE_C': clim_rmse, 'MAE_C': clim_mae},
        f'Persistence ({last_year})': {'RMSE_C': persist_rmse, 'MAE_C': persist_mae},
    }


def train_and_evaluate(images, yms):
    print('\n' + '='*55)
    print('Training ConvLSTM')
    print('='*55)

    images_norm, mu, sig = normalise(images)
    X, y = make_sequences(images_norm)

    # Target month indices (month after the sequence)
    target_yms = [yms[i + SEQ_LEN] for i in range(len(X))]

    # Split: train on pre-policy, test on post-policy
    train_mask = [ym < POLICY_YM for ym in target_yms]
    test_mask  = [ym >= POLICY_YM for ym in target_yms]

    X_train = X[train_mask];  y_train = y[train_mask]
    X_test  = X[test_mask];   y_test  = y[test_mask]
    yms_test = [ym for ym, m in zip(target_yms, test_mask) if m]

    print(f'  Train samples: {len(X_train)} (pre-policy)')
    print(f'  Test  samples: {len(X_test)}  (post-policy)')

    model = build_model()

    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                monitor='val_loss'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(MODEL_DIR / 'convlstm_model.keras',
                                  save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1,
    )

    # Persist the epoch-level history so Figure 9 can be redrawn later
    # without rerunning model.fit().
    history_df = pd.DataFrame(history.history)
    history_df.index.name = 'epoch'
    history_df.to_csv(HISTORY_CSV)
    with open(HISTORY_JSON, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f'  Saved training history: {HISTORY_CSV}')
    print(f'  Saved training history: {HISTORY_JSON}')

    # ── Predictions in original °C ───────────────────────────
    y_pred_norm = model.predict(X_test)
    y_pred = y_pred_norm * sig[..., np.newaxis] + mu[..., np.newaxis]   # de-normalise
    y_true = y_test      * sig[..., np.newaxis] + mu[..., np.newaxis]

    # ── ConvLSTM metrics ─────────────────────────────────────
    rmse = np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    mae  = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    print(f'\n  ConvLSTM  RMSE : {rmse:.3f} °C')
    print(f'  ConvLSTM  MAE  : {mae:.3f} °C')

    # ── Baseline comparison ───────────────────────────────────
    baselines = compute_baselines(images, yms, yms_test)

    rows = []
    for name, vals in baselines.items():
        rows.append({'model': name, 'RMSE_C': vals['RMSE_C'], 'MAE_C': vals['MAE_C']})
    rows.append({'model': 'ConvLSTM', 'RMSE_C': rmse, 'MAE_C': mae})
    pd.DataFrame(rows).to_csv(TAB_DIR / 'convlstm_metrics.csv', index=False)
    print(f'\n  Baseline comparison saved to convlstm_metrics.csv')

    return model, history, y_true, y_pred, yms_test, mu, sig


def load_saved_history():
    """Load training history from disk when figures are regenerated later."""
    if HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV)
        if 'epoch' in df.columns:
            df = df.drop(columns=['epoch'])
        return df.to_dict(orient='list')
    if HISTORY_JSON.exists():
        with open(HISTORY_JSON) as f:
            return json.load(f)
    return None


# ════════════════════════════════════════════════════════════
# STEP 4  —  Spatial residual map (cooling fingerprint)
# ════════════════════════════════════════════════════════════

def plot_residual_map(y_true, y_pred, yms_test):
    """
    Mean residual = mean(predicted) − mean(observed) over post-policy period.
    Positive residual → model predicted warmer than observed
    → actual cooling beyond what pre-policy dynamics would produce
    → evidence of policy-driven cooling
    """
    mean_pred = y_pred.mean(axis=0).squeeze()
    mean_true = y_true.mean(axis=0).squeeze()
    residual  = mean_pred - mean_true   # positive = extra cooling
    true_std  = y_true.std(axis=0).squeeze()
    pred_std  = y_pred.std(axis=0).squeeze()

    # Filled background cells are effectively constant through time.
    # For panel (b), render those cells using the observed background so
    # the two temperature panels share the same visual backdrop.
    background_mask = (true_std < 1e-6) & (pred_std < 1e-6)
    mean_pred_plot = mean_pred.copy()
    mean_pred_plot[background_mask] = mean_true[background_mask]

    temp_vmin = min(mean_true.min(), mean_pred.min())
    temp_vmax = max(mean_true.max(), mean_pred.max())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: mean observed LST
    im0 = axes[0].imshow(mean_true, cmap='RdYlBu_r',
                         vmin=temp_vmin, vmax=temp_vmax)
    axes[0].set_title('(a)  Mean observed LST\n(post-policy)',
                      fontsize=10, pad=6)
    plt.colorbar(im0, ax=axes[0], label='°C', shrink=0.8)

    # Panel 2: mean predicted (counterfactual)
    im1 = axes[1].imshow(mean_pred_plot, cmap='RdYlBu_r',
                         vmin=temp_vmin, vmax=temp_vmax)
    axes[1].set_title('(b)  Mean predicted LST\n(counterfactual)',
                      fontsize=10, pad=6)
    plt.colorbar(im1, ax=axes[1], label='°C', shrink=0.8)

    # Panel 3: residual (cooling map)
    lim = np.abs(residual).max()
    im2 = axes[2].imshow(residual, cmap='RdBu',
                          vmin=-lim, vmax=lim)
    axes[2].set_title('(c)  Cooling residual\n(predicted − observed)',
                       fontsize=10, pad=6)
    plt.colorbar(im2, ax=axes[2], label='°C (+ = more cooling)', shrink=0.8)

    for ax in axes:
        ax.axis('off')

    fig.suptitle('Figure 8  —  ConvLSTM spatial cooling fingerprint\n'
                 '(positive residual = observed cooler than pre-policy dynamics predict)',
                 fontsize=11, y=1.005)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(FIG_DIR / 'fig8_convlstm_residuals.png')
    plt.close()
    print('  Saved fig8_convlstm_residuals.png')


# ════════════════════════════════════════════════════════════
# STEP 5  —  Time-series comparison (spatial mean)
# ════════════════════════════════════════════════════════════

def plot_timeseries(y_true, y_pred, yms_test, history=None):
    true_ts = y_true.mean(axis=(1, 2, 3))
    pred_ts = y_pred.mean(axis=(1, 2, 3))
    dates   = pd.to_datetime(yms_test)
    history_dict = None
    if history is not None:
        history_dict = history.history
    else:
        history_dict = load_saved_history()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    # Panel 1: observed vs predicted
    ax = axes[0]
    ax.plot(dates, true_ts, color='#333', lw=2,   label='Observed (post-policy)')
    ax.plot(dates, pred_ts, color='#E8593C', lw=2,
            ls='--', label='ConvLSTM counterfactual')
    ax.fill_between(dates, pred_ts, true_ts,
                    where=(true_ts < pred_ts),
                    alpha=0.25, color='#3B6D11', label='Cooling (obs < predicted)')
    ax.set_ylabel('Spatial mean LST (°C)')
    ax.set_title('(a)  ConvLSTM predicted vs observed (post-policy)',
                 fontsize=10)
    ax.legend(fontsize=9)

    # Panel 2: training loss curve
    ax = axes[1]
    if history_dict and 'loss' in history_dict and 'val_loss' in history_dict:
        ax.plot(history_dict['loss'],     label='Training loss')
        ax.plot(history_dict['val_loss'], label='Validation loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE loss')
        ax.set_title('(b)  Training curve', fontsize=10)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.56, 'Training curve unavailable', ha='center', va='center',
                fontsize=11)
        ax.text(0.5, 0.41, 'No saved History object or history file found.',
                ha='center', va='center', fontsize=10, color='#555')
        ax.set_axis_off()

    fig.suptitle('Figure 9  —  ConvLSTM training and post-policy performance',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig9_convlstm_timeseries.png')
    plt.close()
    print('  Saved fig9_convlstm_timeseries.png')


# ── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading Landsat tiles …')
    images, yms = load_tiles()

    model, history, y_true, y_pred, yms_test, mu, sig = \
        train_and_evaluate(images, yms)

    print('\nGenerating output figures …')
    plot_residual_map(y_true, y_pred, yms_test)
    plot_timeseries(y_true, y_pred, yms_test, history)

    print(f'\n✅  Script 06 complete.')
    print(f'   Model saved : {MODEL_DIR / "convlstm_model.keras"}')
    print(f'   Figures     : {FIG_DIR}')
    print(f'   Metrics     : {TAB_DIR / "convlstm_metrics.csv"}')
