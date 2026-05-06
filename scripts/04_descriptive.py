"""
Chengdu Urban Cooling Project — Script 04
Purpose : Descriptive analysis and publication-quality trend figures.

Outputs (saved to  outputs/figures/):
  fig1_lst_trend.png        — Urban & rural LST time series + policy break
  fig2_uhi_trend.png        — UHI intensity time series
  fig3_ndvi_trend.png       — NDVI urban trend
  fig4_seasonal_heatmap.png — Monthly LST heatmap (year × month)
  fig5_weather_controls.png — ERA5 controls overview

Run: python scripts/04_descriptive.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────
PANEL   = Path('data/panel/chengdu_monthly_panel.csv')
FIG_DIR = Path('outputs/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.linewidth'   : 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})

POLICY_YEAR = 2022
POLICY_COLOR = '#E8593C'
URBAN_COLOR  = '#D85A30'
RURAL_COLOR  = '#378ADD'
NDVI_COLOR   = '#3B6D11'
UHI_COLOR    = '#7F77DD'


def load():
    df = pd.read_csv(PANEL)
    df['date'] = pd.to_datetime(df['ym'])
    return df


def add_policy_line(ax, df, label=True):
    """Draw a vertical dashed line at January 2022."""
    policy_date = pd.Timestamp('2022-01-01')
    ax.axvline(policy_date, color=POLICY_COLOR, lw=1.4,
               ls='--', alpha=0.85, zorder=3)
    if label:
        ax.text(pd.Timestamp('2022-03-01'),
                ax.get_ylim()[1] * 0.97,
                'Policy\nonset', color=POLICY_COLOR,
                fontsize=9, va='top')


def add_shading(ax, df):
    """Shade pre- and post-policy periods."""
    pre_end  = pd.Timestamp('2022-01-01')
    post_end = df['date'].max()
    ax.axvspan(df['date'].min(), pre_end,
               alpha=0.04, color='steelblue', label='Pre-policy')
    ax.axvspan(pre_end, post_end,
               alpha=0.06, color=POLICY_COLOR, label='Post-policy')


def rolling_annual(series, window=12):
    return series.rolling(window, center=True, min_periods=6).mean()


# ── Figure 1: LST urban vs rural ────────────────────────────
def fig1_lst(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    add_shading(ax, df)

    ax.plot(df['date'], df['LST_urban_C'],
            color=URBAN_COLOR, lw=0.7, alpha=0.4)
    ax.plot(df['date'], rolling_annual(df['LST_urban_C']),
            color=URBAN_COLOR, lw=2, label='Urban LST (12-month rolling mean)')

    ax.plot(df['date'], df['LST_rural_C'],
            color=RURAL_COLOR, lw=0.7, alpha=0.4)
    ax.plot(df['date'], rolling_annual(df['LST_rural_C']),
            color=RURAL_COLOR, lw=2, label='Rural LST (12-month rolling mean)')

    add_policy_line(ax, df)

    ax.set_xlabel('Date')
    ax.set_ylabel('Daytime Land Surface Temperature (°C)')
    ax.set_title('Figure 1  —  Urban and rural LST, Chengdu 2016–2025',
                 fontsize=12, pad=10)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_lst_trend.png')
    plt.close()
    print('  Saved fig1_lst_trend.png')


# ── Figure 2: UHI intensity ──────────────────────────────────
def fig2_uhi(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    add_shading(ax, df)

    ax.plot(df['date'], df['UHI_intensity'],
            color=UHI_COLOR, lw=0.7, alpha=0.4)
    ax.plot(df['date'], rolling_annual(df['UHI_intensity']),
            color=UHI_COLOR, lw=2.2, label='UHI intensity')
    ax.axhline(0, color='grey', lw=0.8, ls=':')

    # Annotate pre/post means
    pre  = df[df['year'] < POLICY_YEAR]['UHI_intensity'].mean()
    post = df[df['year'] >= POLICY_YEAR]['UHI_intensity'].mean()
    ax.annotate(f'Pre mean\n{pre:.2f} °C',
                xy=(pd.Timestamp('2019-01-01'), pre),
                fontsize=8.5, color='#444', ha='center')
    ax.annotate(f'Post mean\n{post:.2f} °C',
                xy=(pd.Timestamp('2023-06-01'), post),
                fontsize=8.5, color=POLICY_COLOR, ha='center')

    add_policy_line(ax, df)
    ax.set_xlabel('Date')
    ax.set_ylabel('UHI Intensity (urban − rural LST, °C)')
    ax.set_title('Figure 2  —  Urban Heat Island intensity, Chengdu 2016–2025',
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_uhi_trend.png')
    plt.close()
    print('  Saved fig2_uhi_trend.png')


# ── Figure 3: NDVI urban ────────────────────────────────────
def fig3_ndvi(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    add_shading(ax, df)

    ax.plot(df['date'], df['NDVI_urban'],
            color=NDVI_COLOR, lw=0.7, alpha=0.4)
    ax.plot(df['date'], rolling_annual(df['NDVI_urban']),
            color=NDVI_COLOR, lw=2.2, label='Urban NDVI')
    ax.plot(df['date'], rolling_annual(df['NDVI_rural']),
            color='#888', lw=1.5, ls='--', label='Rural NDVI (reference)')

    add_policy_line(ax, df)
    ax.set_xlabel('Date')
    ax.set_ylabel('NDVI')
    ax.set_title('Figure 3  —  Vegetation index (NDVI), Chengdu 2016–2025',
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_ndvi_trend.png')
    plt.close()
    print('  Saved fig3_ndvi_trend.png')


# ── Figure 4: Monthly LST heatmap ───────────────────────────
def fig4_heatmap(df):
    pivot = df.pivot_table(index='year', columns='month',
                           values='LST_urban_C', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlBu_r',
                   vmin=pivot.values[~np.isnan(pivot.values)].min(),
                   vmax=pivot.values[~np.isnan(pivot.values)].max())

    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Policy onset line between 2021 and 2022 rows
    policy_row = list(pivot.index).index(POLICY_YEAR) - 0.5
    ax.axhline(policy_row, color=POLICY_COLOR, lw=2, ls='--')
    ax.text(11.6, policy_row - 0.1, 'Policy\nonset',
            color=POLICY_COLOR, fontsize=8, va='center')

    plt.colorbar(im, ax=ax, label='Mean daytime LST (°C)', shrink=0.8)
    ax.set_title('Figure 4  —  Monthly LST heatmap (urban core)',
                 fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_seasonal_heatmap.png')
    plt.close()
    print('  Saved fig4_seasonal_heatmap.png')


# ── Figure 5: Weather controls ──────────────────────────────
def fig5_weather(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for ax, col, label, color in zip(
        axes,
        ['temp_2m_C', 'precip_mm', 'solar_rad_Wm2'],
        ['2 m air temperature (°C)', 'Precipitation (mm)', 'Solar radiation (W m⁻²)'],
        ['#D85A30', '#378ADD', '#EF9F27']
    ):
        ax.plot(df['date'], df[col], color=color, lw=0.8, alpha=0.5)
        ax.plot(df['date'], rolling_annual(df[col]),
                color=color, lw=1.8)
        ax.axvline(pd.Timestamp('2022-01-01'),
                   color=POLICY_COLOR, lw=1.2, ls='--', alpha=0.7)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=9)

    axes[0].set_title('Figure 5  —  ERA5 weather control variables, Chengdu 2016–2025',
                      fontsize=12, pad=10)
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_weather_controls.png')
    plt.close()
    print('  Saved fig5_weather_controls.png')


# ── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load()
    print(f'Panel loaded: {len(df)} rows\n')
    print('Generating figures …')
    fig1_lst(df)
    fig2_uhi(df)
    fig3_ndvi(df)
    fig4_heatmap(df)
    fig5_weather(df)
    print(f'\n✅  All figures saved to {FIG_DIR}')
    print('   Next: python scripts/05_causal_impact.py')
