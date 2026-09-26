"""E2: bifunctional-gap verdict (Workstream E, task 2).

QUESTION: is the empty HER x ORR dual-optimal intersection a physical fact or
a resolution limit of the model?

DATA CONVENTION (declared): LABEL-based. We use measured (comp,structure,facet)
rows shared between the H* and OH* datasets (n~229), not model predictions, so
the scaling line reflects physics rather than model bias. Demo-level free-energy
corrections identical to Workstream C-1 dual_volcano.py:
  dG_H*  = E_ads(H*)  + 0.24 eV
  dG_OH* = E_ads(OH*) + 3.2 eV
(HER x OER uses dG_O* = E_ads(O*) + 3.2 eV analogously, demo-level.)

METHOD: Theil-Sen fit dG_OH = a*dG_H + b; minimum distance from the scaling
line to the ideal point (0,0) is d_min = |b|/sqrt(a^2+1). Compare d_min with
the model resolution 0.11 eV (v3 OOF MAE 0.1134). If d_min >> resolution the
empty intersection is a physical fact (for this dataset's coverage).

SCOPE QUALIFIER (applies to every conclusion here): ordered fcc intermetallic
surfaces, composition-level granularity, demo-level free-energy corrections.

Outputs: e2_gap_verdict.json, e2_gap.png (scaling line + ideal point +
d_min annotation + 0.11 eV resolution circle). Idempotent.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data_processed', 'multiads_dataset.csv')
RESOLUTION = 0.11  # eV, v3 OOF MAE 0.1134 rounded
KEY = ['comp', 'structure', 'facet']
SCOPE = ('ordered fcc intermetallic surfaces, composition-level granularity, '
         'demo-level free-energy corrections (+0.24 eV for H*, +3.2 eV for '
         'OH*/O*)')


def fit_pair(df, ads_x, ads_y, shift_x, shift_y):
    gx = df[df.adsorbate == ads_x][KEY + ['energy_eV']]
    gy = df[df.adsorbate == ads_y][KEY + ['energy_eV']]
    m = gx.merge(gy, on=KEY, suffixes=('_x', '_y'))
    x = m.energy_eV_x.values + shift_x
    y = m.energy_eV_y.values + shift_y
    a, b, _, _ = theilslopes(y, x)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    d_min = abs(b) / np.sqrt(a ** 2 + 1)
    # nearest measured point to the ideal point, for context
    d_pts = np.sqrt(x ** 2 + y ** 2)
    return dict(x=x, y=y, a=float(a), b=float(b), r2=float(r2),
                d_min=float(d_min), n=len(m),
                nearest_point_dist=float(d_pts.min()))


def verdict(d_min, r2, n, tag='HER x ORR'):
    ratio = d_min / RESOLUTION
    if d_min > 3 * RESOLUTION:
        v = ('PHYSICAL GAP: d_min = %.3f eV is %.1fx the model resolution '
             '(%.2f eV); the empty %s intersection is NOT a resolution '
             'limit -- it is a physical fact within the dataset coverage '
             '(%s).') % (d_min, ratio, RESOLUTION, tag, SCOPE)
    elif d_min > RESOLUTION:
        v = ('BORDERLINE: d_min = %.3f eV is only %.1fx the model resolution; '
             'gap likely physical but close to resolution limit (%s).'
             % (d_min, ratio, SCOPE))
    else:
        v = ('RESOLUTION-LIMITED: d_min = %.3f eV <= resolution %.2f eV; '
             'cannot rule out dual-optimal compositions (%s).'
             % (d_min, RESOLUTION, SCOPE))
    return v, ratio


def main():
    df = pd.read_csv(DATA)
    out = {'scope_qualifier': SCOPE, 'data_convention': 'label-based '
           '(measured shared compositions, no model predictions)',
           'resolution_eV': RESOLUTION}

    her_orr = fit_pair(df, 'H', 'OH', 0.24, 3.2)
    v, ratio = verdict(her_orr['d_min'], her_orr['r2'], her_orr['n'])
    out['HER_x_ORR'] = dict(scaling_line='dG_OH = %.4f*dG_H + %.4f'
                            % (her_orr['a'], her_orr['b']),
                            slope=round(her_orr['a'], 4),
                            intercept=round(her_orr['b'], 4),
                            r2=round(her_orr['r2'], 4), n=her_orr['n'],
                            d_min_eV=round(her_orr['d_min'], 4),
                            d_min_over_resolution=round(ratio, 1),
                            nearest_measured_point_dist_eV=round(
                                her_orr['nearest_point_dist'], 4),
                            verdict=v)
    print(v)

    try:
        her_oer = fit_pair(df, 'H', 'O', 0.24, 3.2)
        v2, ratio2 = verdict(her_oer['d_min'], her_oer['r2'], her_oer['n'], tag='HER x OER')
        out['HER_x_OER'] = dict(scaling_line='dG_O = %.4f*dG_H + %.4f'
                                % (her_oer['a'], her_oer['b']),
                                slope=round(her_oer['a'], 4),
                                intercept=round(her_oer['b'], 4),
                                r2=round(her_oer['r2'], 4), n=her_oer['n'],
                                d_min_eV=round(her_oer['d_min'], 4),
                                d_min_over_resolution=round(ratio2, 1),
                                verdict=v2)
        print(v2)
    except Exception as e:
        out['HER_x_OER'] = dict(error=str(e))

    # --- figure: HER x ORR bifunctional gap ---
    x, y, a, b = (her_orr[k] for k in ('x', 'y', 'a', 'b'))
    d_min = her_orr['d_min']
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=18, alpha=0.5, color='#DFB27E', edgecolors='none',
               label='measured shared comps (n=%d)' % len(x))
    xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 50)
    ax.plot(xs, a * xs + b, color='#B35C24', lw=2,
            label=r'scaling line: $\Delta G_{OH}$=%.2f$\cdot\Delta G_H$+%.2f'
                  % (a, b))
    ax.scatter([0], [0], s=180, marker='*', color='#b5522d',
               edgecolors='#7A4A2B', zorder=5, label='ideal point (0,0)')
    ax.add_patch(plt.Circle((0, 0), RESOLUTION, fill=False, ls='--',
                            color='#8a7b4f', lw=1.4,
                            label='model resolution 0.11 eV'))
    # d_min segment: perpendicular foot from origin to line
    xf = -a * b / (a ** 2 + 1)
    yf = b / (a ** 2 + 1)
    ax.plot([0, xf], [0, yf], color='#7A4A2B', lw=1.6, ls=':')
    mx, my = xf / 2, yf / 2
    ax.annotate(r'$d_{min}$=%.2f eV (%.0fx resolution)' % (d_min, d_min / RESOLUTION),
                xy=(mx, my), xytext=(-1.55, 1.75), fontsize=10,
                color='#7A4A2B',
                arrowprops=dict(arrowstyle='->', color='#7A4A2B'))
    ax.set_xlabel(r'$\Delta G_{H^*}$ (eV)  [HER, demo correction]')
    ax.set_ylabel(r'$\Delta G_{OH^*}$ (eV)  [ORR, demo correction]')
    ax.set_title('Bifunctional gap: scaling line vs ideal point\n'
                 '(ordered fcc surfaces, composition-level, label-based)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'e2_gap.png'), dpi=300)

    with open(os.path.join(HERE, 'e2_gap_verdict.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out['HER_x_ORR'].items()
                      if k != 'verdict'}, indent=2))


if __name__ == '__main__':
    main()
