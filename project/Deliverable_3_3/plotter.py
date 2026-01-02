from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from LinearMPC.MPCVelControl import MPCVelControl
import os

def start_fig(name: str) -> None:
    FIGS_SIZES = (10, 5)
    plt.close('all')
    plt.figure(name, FIGS_SIZES)

def file_name(name: str) -> str:
    return name \
        .lower() \
        .replace(' ', '-') \
        .replace(',', '') \
        .replace('°', '') \
        .replace('/', '')

def save_fig(name: str) -> None:
    PLOT_DIR = 'plots/'
    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(PLOT_DIR + file_name(name), dpi=200)
    plt.show()

def plot_x_y_terminal_set(name: str, mpc: MPCVelControl) -> None:

    start_fig(name)
    OPACITY = 0.1
    X_COLOR = 'teal'
    Y_COLOR = 'orange'

    # Plot first dimensions pair

    ax = plt.subplot(221)
    mpc.mpc_x.O_inf.projection((1, 0)).plot(ax, color=X_COLOR, opacity=OPACITY)
    plt.ylabel(r'$\omega_y \quad (\frac{\text{rad}}{\text{s}})$')
    plt.grid()

    ax = plt.subplot(222)
    mpc.mpc_y.O_inf.projection((1, 0)).plot(ax, color=Y_COLOR, opacity=OPACITY)
    plt.ylabel(r'$\omega_x \quad (\frac{\text{rad}}{\text{s}})$')
    plt.grid()

    # Plot second dimensions pair

    ax = plt.subplot(223)
    mpc.mpc_x.O_inf.projection((1, 2)).plot(ax, color=X_COLOR, opacity=OPACITY)
    plt.xlabel(r'$\beta \quad (rad)$')
    plt.ylabel(r'$v_x \quad (\frac{\text{m}}{\text{s}})$')
    plt.grid()

    ax = plt.subplot(224)
    mpc.mpc_y.O_inf.projection((1, 2)).plot(ax, color=Y_COLOR, opacity=OPACITY)
    plt.xlabel(r'$\alpha \quad (\text{rad})$')
    plt.ylabel(r'$v_y \quad (\frac{\text{m}}{\text{s}})$')
    plt.grid()

    save_fig(name)


def enforce_min_yrange(ax: Axes, min_range: float = 2):
    y0, y1 = ax.get_ylim()
    rng = y1 - y0
    if rng < min_range:
        mid = 0.5 * (y0 + y1)
        half = 0.5 * min_range
        ax.set_ylim(mid - half, mid + half)

def plot_traj(name: str, x: np.ndarray, u: np.ndarray, t: np.ndarray, ref: np.ndarray | None = None) -> None:

    start_fig(name)

    # Color palette

    X_COLOR = 'xkcd:red'
    Y_COLOR = 'xkcd:green'
    Z_COLOR = 'xkcd:blue'
    X_REF_COLOR = 'xkcd:darkish red'
    Y_REF_COLOR = 'xkcd:darkish green'
    Z_REF_COLOR = 'xkcd:darkish blue'
    PRIMARY_COLOR = 'orange'
    SECONDARY_COLOR = 'teal'
    LIMITS_COLOR = 'xkcd:slate'

    PRIMARY_STYLE = (0, (2.5, 3))
    SECONDARY_STYLE = (2.75, (2.5, 3))
    LIMITS_STYLE = 'dashed'
    REF_STYLE = lambda i : (3 * i, (3, 6))
    LIMITS_WIDTH = 2.5
    REF_WIDTH = 1.0

    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'serif'

    # Linear states

    ax = plt.subplot(231)
    if ref is not None:
        ax.plot(t[:-1], ref[6, :-1], color=X_REF_COLOR, linestyle=REF_STYLE(0), linewidth=REF_WIDTH)
        ax.plot(t[:-1], ref[7, :-1], color=Y_REF_COLOR, linestyle=REF_STYLE(1), linewidth=REF_WIDTH)
        ax.plot(t[:-1], ref[8, :-1], color=Z_REF_COLOR, linestyle=REF_STYLE(2), linewidth=REF_WIDTH)
    ax.plot(t[:-1], x[6, :-1], color=X_COLOR, linestyle=PRIMARY_STYLE, label=r'$v_x$')
    ax.plot(t[:-1], x[7, :-1], color=Y_COLOR, linestyle=SECONDARY_STYLE, label=r'$v_y$')
    ax.plot(t[:-1], x[8, :-1], color=Z_COLOR, label=r'$v_z$')
    ax.set_ylabel(r'$\boldsymbol{v} \quad (\frac{\text{m}}{\text{s}})$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    ax = plt.subplot(234)
    if ref is not None:
        ax.plot(t[:-1], ref[9, :-1], color=X_REF_COLOR, linestyle=REF_STYLE(0), linewidth=REF_WIDTH)
        ax.plot(t[:-1], ref[10, :-1], color=Y_REF_COLOR, linestyle=REF_STYLE(1), linewidth=REF_WIDTH)
        ax.plot(t[:-1], ref[11, :-1], color=Z_REF_COLOR, linestyle=REF_STYLE(2), linewidth=REF_WIDTH)
    ax.plot(t[:-1], x[9, :-1], color=X_COLOR, linestyle=PRIMARY_STYLE, label=r'$p_x$')
    ax.plot(t[:-1], x[10, :-1], color=Y_COLOR, linestyle=SECONDARY_STYLE, label=r'$p_y$')
    ax.plot(t[:-1], x[11, :-1], color=Z_COLOR, label=r'$p_z$')
    ax.set_xlabel(r'$t \quad (s)$')
    ax.set_ylabel(r'$\boldsymbol{p} \quad (\text{m})$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    # Angular states

    ax = plt.subplot(232)
    if ref is not None:
        ax.plot(t[:-1], np.rad2deg(ref[0, :-1]), color=X_REF_COLOR, linestyle=REF_STYLE(0), linewidth=REF_WIDTH)
        ax.plot(t[:-1], np.rad2deg(ref[1, :-1]), color=Y_REF_COLOR, linestyle=REF_STYLE(1), linewidth=REF_WIDTH)
        ax.plot(t[:-1], np.rad2deg(ref[2, :-1]), color=Z_REF_COLOR, linestyle=REF_STYLE(2), linewidth=REF_WIDTH)
    ax.plot(t[:-1], np.rad2deg(x[0, :-1]), color=X_COLOR, label=r'$\omega_x$')
    ax.plot(t[:-1], np.rad2deg(x[1, :-1]), color=Y_COLOR, label=r'$\omega_y$')
    ax.plot(t[:-1], np.rad2deg(x[2, :-1]), color=Z_COLOR, label=r'$\omega_z$')
    ax.set_ylabel(r'$\boldsymbol{\omega} \quad (\frac{^\circ}{\text{s}})$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    ax = plt.subplot(235)
    ax.axhline(10, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.axhline(-10, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    if ref is not None:
        ax.plot(t[:-1], np.rad2deg(ref[3, :-1]), color=X_REF_COLOR, linestyle=REF_STYLE(0), linewidth=REF_WIDTH)
        ax.plot(t[:-1], np.rad2deg(ref[4, :-1]), color=Y_REF_COLOR, linestyle=REF_STYLE(1), linewidth=REF_WIDTH)
        ax.plot(t[:-1], np.rad2deg(ref[5, :-1]), color=Z_REF_COLOR, linestyle=REF_STYLE(2), linewidth=REF_WIDTH)
    ax.plot(t[:-1], np.rad2deg(x[3, :-1]), color=X_COLOR, label=r'$\alpha$')
    ax.plot(t[:-1], np.rad2deg(x[4, :-1]), color=Y_COLOR, label=r'$\beta$')
    ax.plot(t[:-1], np.rad2deg(x[5, :-1]), color=Z_COLOR, label=r'$\gamma$')
    ax.set_yticks(np.arange(-10, 30 + 1, 10))
    ax.set_xlabel(r'$t \quad (s)$')
    ax.set_ylabel(r'$\boldsymbol{\varphi} \quad (^\circ)$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    # Inputs

    ax = plt.subplot(233)
    ax.axhline(15, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.axhline(-15, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.plot(t[:-1], np.rad2deg(u[0, :]), color=PRIMARY_COLOR, linestyle=PRIMARY_STYLE, label=r'$\delta_1$')
    ax.plot(t[:-1], np.rad2deg(u[1, :]), color=SECONDARY_COLOR, linestyle=SECONDARY_STYLE, label=r'$\delta_2$')
    ax.set_yticks(np.arange(-15, 15 + 1, 7.5))
    ax.set_ylabel(r'$\boldsymbol{\delta} \quad (^\circ)$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    ax = plt.subplot(236)
    ax.axhline(40, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.axhline(80, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.axhline(-20, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.axhline(20, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.plot(t[:-1], u[2, :], color=PRIMARY_COLOR, label=r'$P_\text{avg}$')
    ax.plot(t[:-1], u[3, :], color=SECONDARY_COLOR, label=r'$P_\text{diff}$')
    ax.set_yticks(np.arange(-20, 80 + 1, 20))
    ax.set_xlabel(r'$t \quad (s)$')
    ax.set_ylabel(r'$\boldsymbol{P} \quad (\%)$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    save_fig(name)
