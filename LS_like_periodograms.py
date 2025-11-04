"""
    Created on 2025, Oct 14.

    Description: Implementation of Stacked Periodograms, along with compilation of Lomb-Scargle algorithms.

    @authors:   Cleber Silva <clebersilva@fisica.ufc.br> [1, 3]
                Eder Martioli <emartioli@lna.br> [2, 3]

    [1] Stellar Team - Universidade Federal do Ceará, Brazil.
    [2] Laboratório Nacional de Astrofísica, Brazil.
    [3] MALTED research group, Brazil.

    **References:**

    A. Mortier, A. Collier Cameron. A&A 601 A110 (2017) - DOI: https://doi.org/10.1051/0004-6361/201630201

    A. Mortier, et al. A&A, 573 (2015) A101 - DOI: https://doi.org/10.1051/0004-6361/201424908

    Jacob T. VanderPlas 2018 ApJS 236 16 - DOI: https://doi.org/10.3847/1538-4365/aab766
"""

import importlib.util
import subprocess
import sys

pkg = "tqdm"
if importlib.util.find_spec(pkg) is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from tqdm import tqdm


def select_N_time_series_points(x, y, dy, N, mode='chronological', idxs_mask=None):
    x = np.array(x)
    y = np.array(y)
    dy = np.array(dy)

    if mode == 'chronological':
        x_cut = x[:N]
        y_cut = y[:N]
        dy_cut = dy[:N]
    
    elif mode == 'random':
        idxs = np.arange(len(x))

        if type(idxs_mask) != type(None):
            mask_len = len(idxs_mask)
            N -= mask_len
            idxs = idxs[~np.isin(idxs, idxs_mask)]
            idxs_mask_new = np.random.choice(idxs, N, replace=False)
            idxs_mask = np.concatenate([idxs_mask, idxs_mask_new])
        else:
            idxs_mask = np.random.choice(idxs, N, replace=False)

        idxs_mask.sort()

        x_cut = x[idxs_mask]
        y_cut = y[idxs_mask]
        dy_cut = dy[idxs_mask]

    return x_cut, y_cut, dy_cut, idxs_mask



def plot_curve_and_periodogram(x, y, dy, num_points, periodogram_type='GLS',
                               spectrum_yscale='linear', mode='chronological',
                               old_idxs_mask=None):
    algorithm = {
        'GLS' : GLS,
        'BGLS' : BGLS
    }

    t_cut, y_cut, dy_cut, idxs_masks = select_N_time_series_points(x, y, dy,
                                                                   N=num_points, mode=mode,
                                                                   idxs_mask=old_idxs_mask)
    result = algorithm[periodogram_type](t_cut, y_cut, dy_cut, fmax=1)

    periods = 1/result['freq']
    power = result['power'] - min(result['power'])
    best_P = result['best_period']

    output = {
        't_cut' : t_cut,
        'y_cut' : y_cut,
        'dy_cut' : dy_cut,
        'idxs_masks' : idxs_masks,
        'periods' : periods,
        'power' : power,
        'best_P' : best_P
    }

    # Plot
    fontsize = 13
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ## Time series
    if type(old_idxs_mask) != type(None):
        new_idxs_mask = idxs_masks[~np.isin(idxs_masks, old_idxs_mask)]
        axes[0].errorbar(x[old_idxs_mask], y[old_idxs_mask], dy[old_idxs_mask],
                         fmt='o', color='k', alpha=0.6)
        axes[0].errorbar(x[new_idxs_mask], y[new_idxs_mask], dy[new_idxs_mask],
                         fmt='o', color='red', alpha=0.6, label='New points')
        axes[0].legend(loc='lower left')
    else:
        axes[0].errorbar(t_cut, y_cut, dy_cut, fmt='o', color='k', alpha=0.6)
    axes[0].set_xlabel("Time [eMJD]", fontsize = fontsize)
    axes[0].set_ylabel("RV [m/s]", fontsize = fontsize)
    axes[0].set_title(f"Time Series: First {num_points} Data Points", fontsize = fontsize+5)
    axes[0].tick_params(axis='both', labelsize=12)

    ## Periodogram power over period by
    axes[1].plot(periods, power, color='grey')
    axes[1].axvline(best_P, ls='--', color='k', alpha=0.7, label=f'Best period: {best_P:.2f} days')
    axes[1].set_xlabel("Period [days]", fontsize = fontsize)
    axes[1].set_ylabel("Signal Power", fontsize = fontsize)
    axes[1].set_title("Corresponding Power Spectrum", fontsize = fontsize+5)
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].set_yscale(spectrum_yscale)
    # axes[1].set_xlim(0, 10)
    axes[1].legend(fontsize = fontsize)

    plt.show()

    return output


def get_frequency_grid(fmin = 0.03, fmax = 2.0, num_freqs = 5000):
    return np.linspace(fmin, fmax, num_freqs)


def GLS(time, y, dy, fmin = 0.03, fmax = 2.0, num_freqs = 5000):
    """
    Computes the Generalized Lomb-Scargle Periodogram.
    """

    freq = get_frequency_grid(fmin = fmin, fmax = fmax, num_freqs = num_freqs)

    ls = LombScargle(time, y, dy, center_data=False)
    power_gls = ls.power(freq)

    best_f_gls = freq[np.argmax(power_gls)]
    best_P_gls = 1.0 / best_f_gls
    
    result = {
        'freq' : freq,
        'power' : power_gls,
        'best_period' : best_P_gls
    }

    return result


def BGLS(time, y, dy, fmin = 0.03, fmax = 2.0, num_freqs = 5000):
    """
    Computes the Bayesian Generalized Lomb-Scargle following Mortier et al. 2015.
    """

    freq = get_frequency_grid(fmin = fmin, fmax = fmax, num_freqs = num_freqs)
    omega_array = 2 * np.pi * freq

    logEv_array = []

    for omega in omega_array:
        w = 1.0 / (dy**2)
        Wy = w * y
        yWy = np.dot(y, Wy)

        c = np.cos(omega * time)
        s = np.sin(omega * time)

        X = np.column_stack([c, s, np.ones_like(time)])
        
        XtW = (X.T * w)
        XtWX = XtW @ X
        b = XtW @ y

        ridge = 1e-12
        XtWX[np.diag_indices_from(XtWX)] += ridge

        # Solve and determinants
        try:
            L = np.linalg.cholesky(XtWX)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            z = np.linalg.solve(L, b)
            u = np.linalg.solve(L.T, z)
            quad = np.dot(b, u)  # b^T (XtWX)^(-1) b
        except np.linalg.LinAlgError:
            # Fallback to generic inverse if needed
            invXtWX = np.linalg.pinv(XtWX)
            sign, logdet = np.linalg.slogdet(XtWX)
            if sign <= 0:
                # extremely degenerate, so return a very small log-evidence
                return -1e300
            quad = b @ (invXtWX @ b)

        log_evidence = -0.5 * (yWy - quad) - 0.5 * logdet

        logEv_array.append(log_evidence)
    
    logEv_array = np.array(logEv_array)
    logEv_array -= logEv_array.max()
    post = np.exp(logEv_array)

    # Normalizing by the trapezoidal integral over frequency so it sums to 1 in the continuous sense
    # Becareful here whith the numpy version!!!
    norm = np.trapz(post, freq)
    post /= norm if norm > 0 else 1.0

    best_f_bgls = freq[np.argmax(post)]
    best_P_bgls = 1.0 / best_f_bgls

    result = {
        'freq' : freq,
        'power' : post,
        'best_period' : best_P_bgls
    }

    return result


def stacked_periodogram(t, y, dy, N_min, periodogram_type='GLS',
                        p_min = 0.5, p_max = 50, num_periods = 5000,
                        mode='chronological', idxs_masks=None):
    fmin = 1/p_max
    fmax = 1/p_min

    algorithm = {
        'GLS' : GLS,
        'BGLS' : BGLS
    }
    
    N_max = len(t)

    N_array = []
    periods_array = []
    power_array = []

    for i in tqdm(range(N_max-N_min)):
        current_N = N_min+i
        if mode == 'random with no replacement':
            mode = mode.split(' ')[0]
            idxs_masks = None
        t_cut, y_cut, dy_cut, idxs_masks = select_N_time_series_points(t, y, dy,
                                                                       N=current_N, mode=mode,
                                                                       idxs_mask=idxs_masks)
        bgls_result = algorithm[periodogram_type](t_cut, y_cut, dy_cut,
                                                  fmin = fmin, fmax = fmax,
                                                  num_freqs = num_periods)

        N_array.append(np.zeros(len(bgls_result['freq'])) + current_N)
        periods_array.append(1/bgls_result['freq'])
        power = bgls_result['power'] - min(bgls_result['power']) + 1
        power_array.append(power)

    data = np.stack([N_array, periods_array, power_array], axis=0)

    return data


def plot_stacked_periodogram_heatmap(data, cmap="Reds", vmin=None, vmax=None, norm='linear'):
    """
    Plot heatmap from `data` = np.stack([N_array, periods_array, power_array], axis=0)
    where each entry is a list/array per N having the same number of frequencies.

    x-axis: periods
    y-axis: current_N
    z-axis: power (color)
    """
    N, P, Z = data[0], data[1], data[2]
    N = np.asarray(N)
    P = np.asarray(P)
    Z = np.asarray(Z)

    # Sort columns by period and rows by N, just in case
    col_order = np.argsort(P[0])
    row_order = np.argsort(N[:, 0])

    P = P[row_order][:, col_order]
    N = N[row_order][:, col_order]
    Z = Z[row_order][:, col_order]

    fontsize = 13

    fig, ax = plt.subplots(figsize=(8, 5))

    if norm == 'linear':
        pass
    elif norm == 'log':
        Z = np.log10(Z)
    else:
        raise Exception('Invalid normalization method.')

    im = ax.pcolormesh(P, N, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Power Spectrum Intensity", fontsize=fontsize)

    ax.set_xlabel("Period [days]", fontsize=fontsize)
    ax.set_ylabel("N of Observations", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.show()