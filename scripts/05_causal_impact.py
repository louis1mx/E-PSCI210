"""
Chengdu Urban Cooling Project — Script 05
Purpose : Causal analysis of park-city policy effect on LST

  A. Interrupted Time Series (ITS) regression
     - OLS with month fixed effects + weather controls
     - Estimates level shift and slope change at policy onset
     - Constructs counterfactual: what LST would have been without policy

  B. Pre/post comparison
     - Simple mean comparison before and after 2022
     - Effect size in degrees Celsius

Outputs:
  outputs/figures/fig6_its_regression.png
  outputs/figures/fig7_counterfactual.png
  outputs/tables/its_results.txt
  outputs/tables/policy_effect_summary.csv

Run: python scripts/05_causal_impact.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PANEL   = Path('data/panel/chengdu_monthly_panel.csv')
FIG_DIR = Path('outputs/figures')
TAB_DIR = Path('outputs/tables')
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})


# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════

def load():
    df = pd.read_csv(PANEL)
    df['date'] = pd.to_datetime(df['ym'])

    # Remove confirmed outlier: 2023-10 LST_urban = -44°C (satellite error)
    # Physically impossible for Chengdu; replaced with linear interpolation
    df.loc[df['ym'] == '2023-10', 'LST_urban_C'] = np.nan
    print('Outlier removed: 2023-10 LST_urban set to NaN (was -44°C)')

    # Interpolate missing LST values (cloud gaps + outlier above)
    df['LST_urban_C'] = df['LST_urban_C'].interpolate(
        method='linear', limit_direction='both')
    df['LST_rural_C'] = df['LST_rural_C'].interpolate(
        method='linear', limit_direction='both')
    df['UHI_intensity'] = df['LST_urban_C'] - df['LST_rural_C']

    print(f'Panel loaded: {len(df)} rows')
    print(f'LST urban range: {df["LST_urban_C"].min():.1f} – {df["LST_urban_C"].max():.1f} °C')
    return df


# ════════════════════════════════════════════════════════════
# PART A — ITS Regression
# ════════════════════════════════════════════════════════════

def run_its(df):
    print('\n' + '='*55)
    print('PART A  Interrupted Time Series Regression')
    print('='*55)

    # Create month dummies for seasonal fixed effects
    data = pd.get_dummies(df, columns=['month'], prefix='M', drop_first=True)
    month_cols = sorted([c for c in data.columns if c.startswith('M_')])

    feature_cols = (
        ['time_trend', 'post_policy', 'trend_post',
         'temp_2m_C', 'precip_mm', 'solar_rad_Wm2']
        + month_cols
    )

    # Drop rows with any missing values in needed columns
    data_clean = data.dropna(
        subset=['LST_urban_C'] + feature_cols).copy()

    # Convert all to float (required by statsmodels)
    for c in feature_cols:
        data_clean[c] = data_clean[c].astype(float)
    data_clean['LST_urban_C'] = data_clean['LST_urban_C'].astype(float)

    print(f'Rows used in regression: {len(data_clean)}')

    X = sm.add_constant(data_clean[feature_cols].astype(float))
    y = data_clean['LST_urban_C'].astype(float)
    model = sm.OLS(y, X).fit(cov_type='HC3')

    print(model.summary())

    # Save full results
    with open(TAB_DIR / 'its_results.txt', 'w') as f:
        f.write(model.summary().as_text())
    print(f'\nFull results saved to {TAB_DIR / "its_results.txt"}')

    # ── Key coefficients ─────────────────────────────────────
    print('\nKey policy coefficients:')
    for key in ['post_policy', 'trend_post']:
        b = model.params[key]
        p = model.pvalues[key]
        sig = '***' if p < 0.01 else ('**' if p < 0.05
              else ('*' if p < 0.1 else 'n.s.'))
        print(f'  {key:18s}  β = {b:+.4f}  p = {p:.4f}  {sig}')

    # ── Counterfactual ───────────────────────────────────────
    data_clean['fitted'] = model.fittedvalues

    cf = data_clean.copy()
    cf['post_policy'] = 0.0
    cf['trend_post']  = 0.0
    X_cf = sm.add_constant(cf[feature_cols].astype(float))
    data_clean['counterfactual'] = model.predict(X_cf)
    data_clean['effect'] = (data_clean['counterfactual']
                            - data_clean['LST_urban_C'])

    post = data_clean[data_clean['post_policy'] == 1.0].copy()
    avg_effect = post['effect'].mean()
    print(f'\nAverage cooling effect post-2022: {avg_effect:+.3f} °C')
    print('(positive = observed cooler than counterfactual)')

    return model, data_clean, post


# ════════════════════════════════════════════════════════════
# PART B — Pre/post summary
# ════════════════════════════════════════════════════════════

def pre_post_summary(df):
    print('\n' + '='*55)
    print('PART B  Pre/post summary')
    print('='*55)

    pre  = df[df['year'] < 2022]
    post = df[df['year'] >= 2022]

    rows = []
    for v in ['LST_urban_C', 'LST_rural_C', 'UHI_intensity', 'NDVI_urban']:
        rows.append({
            'variable' : v,
            'pre_mean' : pre[v].mean(),
            'post_mean': post[v].mean(),
            'change'   : post[v].mean() - pre[v].mean(),
        })
    summary = pd.DataFrame(rows)

    print(summary.round(4).to_string(index=False))
    summary.to_csv(TAB_DIR / 'policy_effect_summary.csv', index=False)
    print(f'\nSaved to {TAB_DIR / "policy_effect_summary.csv"}')
    return summary


# ════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════

def fig_its(df, data_clean, post, model):
    """Figure 6 — ITS fitted vs observed with counterfactual."""
    fig, ax = plt.subplots(figsize=(11, 5))

    # Background shading
    ax.axvspan(df['date'].min(), pd.Timestamp('2022-01-01'),
               alpha=0.04, color='steelblue')
    ax.axvspan(pd.Timestamp('2022-01-01'), df['date'].max(),
               alpha=0.06, color='#E8593C')

    # Scatter + fitted line
    ax.scatter(data_clean['date'], data_clean['LST_urban_C'],
               color='#888', s=14, alpha=0.6, label='Observed LST')
    ax.plot(data_clean['date'], data_clean['fitted'],
            color='#333', lw=2, label='ITS fitted values')

    # Counterfactual (post-policy period only)
    ax.plot(post['date'], post['counterfactual'],
            color='#E8593C', lw=2, ls='--',
            label='Counterfactual (no policy)')

    # Fill cooling gap
    ax.fill_between(
        post['date'],
        post['LST_urban_C'],
        post['counterfactual'],
        where=(post['counterfactual'] >= post['LST_urban_C']),
        alpha=0.18, color='#1D9E75',
        label='Estimated cooling')

    # Policy onset line
    ax.axvline(pd.Timestamp('2022-01-01'),
               color='#E8593C', lw=1.4, ls='--', alpha=0.8)

    # Coefficient annotation
    b_level = model.params['post_policy']
    b_slope = model.params['trend_post']
    p_level = model.pvalues['post_policy']
    p_slope = model.pvalues['trend_post']
    ymax = ax.get_ylim()[1]
    ax.text(pd.Timestamp('2022-03-01'), ymax * 0.97,
            f'Level shift: {b_level:+.2f} °C  (p={p_level:.3f})\n'
            f'Slope change: {b_slope:+.4f} °C/month  (p={p_slope:.3f})',
            color='#E8593C', fontsize=9, va='top')

    ax.set_xlabel('Date')
    ax.set_ylabel('Urban daytime LST (°C)')
    ax.set_title(
        'Figure 6  —  Interrupted Time Series: policy effect on urban LST',
        fontsize=12, pad=10)
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_its_regression.png')
    plt.close()
    print('Saved fig6_its_regression.png')


def fig_effect(post):
    """Figure 7 — Monthly cooling effect bar chart."""
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ['#1D9E75' if v >= 0 else '#D85A30' for v in post['effect']]
    ax.bar(post['date'], post['effect'],
           color=colors, width=25, alpha=0.8)
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.axhline(post['effect'].mean(),
               color='#1D9E75', lw=1.5, ls='--',
               label=f'Mean effect: {post["effect"].mean():+.2f} °C')

    ax.set_xlabel('Date')
    ax.set_ylabel('Cooling effect (°C)\n[counterfactual − observed]')
    ax.set_title(
        'Figure 7  —  Monthly policy cooling effect (post-2022)',
        fontsize=12, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig7_counterfactual.png')
    plt.close()
    print('Saved fig7_counterfactual.png')


# ── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load()

    model, data_clean, post = run_its(df)
    pre_post_summary(df)

    print('\nGenerating figures …')
    fig_its(df, data_clean, post, model)
    fig_effect(post)

    print(f'\n✅  Script 05 complete.')
    print(f'   Figures : {FIG_DIR}')
    print(f'   Tables  : {TAB_DIR}')
    print(f'\n   Next: python scripts/06_convlstm.py')
    print(f'   (requires GeoTIFF tiles in data/raw/gee/chengdu_lst_tiles/)')
