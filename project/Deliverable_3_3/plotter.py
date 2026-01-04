from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
from LinearMPC.MPCVelControl import MPCVelControl
import os

def start_fig(name: str) -> Figure:
    FIGS_SIZES = (10, 4.2)
    plt.close('all')
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'serif'
    return plt.figure(name, FIGS_SIZES)

def file_name(name: str) -> str:
    return name \
        .lower() \
        .replace(' ', '-') \
        .replace(',', '') \
        .replace('°', '') \
        .replace('/', '')

def save_fig(name: str) -> None:
    PLOT_DIR = 'plots/'
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(PLOT_DIR + file_name(name), dpi=200)
    plt.show()

def plot_x_y_terminal_set(name: str, mpc: MPCVelControl) -> None:

    start_fig(name)
    OPACITY = 0.7
    X_COLOR = 'xkcd:red'
    Y_COLOR = 'xkcd:green'

    ax = plt.subplot(121, projection='3d', proj_type='ortho')
    ax.view_init(elev=30, azim=45, roll=0)
    mpc.mpc_x.O_inf.plot(ax, color=X_COLOR, opacity=OPACITY, show_edges=True, show_vertices=False)
    ax.set_xlabel(r'$\omega_y \quad (\frac{\text{rad}}{\text{s}})$')
    ax.set_ylabel(r'$\beta \quad (\text{rad})$')
    ax.set_zlabel(r'$x \quad (\text{m})$')
    ax.grid()

    ax = plt.subplot(122, projection='3d', proj_type='ortho')
    ax.view_init(elev=30, azim=-135, roll=0)
    mpc.mpc_y.O_inf.plot(ax, color=Y_COLOR, opacity=OPACITY, show_edges=True, show_vertices=False)
    ax.set_xlabel(r'$\omega_x \quad (\frac{\text{rad}}{\text{s}})$')
    ax.set_ylabel(r'$\alpha \quad (\text{rad})$')
    ax.set_zlabel(r'$y \quad (\text{m})$')
    ax.grid()

    plt.tight_layout(pad=0.5)
    save_fig(name)


def enforce_min_yrange(ax: Axes, min_range: float = 1.5):
    y0, y1 = ax.get_ylim()
    rng = y1 - y0
    if rng < min_range:
        mid = 0.5 * (y0 + y1)
        half = 0.5 * min_range
        ax.set_ylim(mid - half, mid + half)

def plot_traj(name: str, x: np.ndarray, u: np.ndarray, t: np.ndarray, ref: np.ndarray | None = None) -> None:

    fig = start_fig(name)

    # Color palette

    X_COLOR = 'xkcd:red'
    Y_COLOR = 'xkcd:green'
    Z_COLOR = 'xkcd:blue'
    ROLL_COLOR = 'xkcd:golden rod'
    X_REF_COLOR = 'xkcd:darkish red'
    Y_REF_COLOR = 'xkcd:darkish green'
    Z_REF_COLOR = 'xkcd:darkish blue'
    ROLL_REF_COLOR = 'xkcd:dark gold'
    LIMITS_COLOR = 'xkcd:slate'

    PRIMARY_STYLE = (0, (2.5, 3))
    SECONDARY_STYLE = (2.75, (2.5, 3))
    LIMITS_STYLE = 'dashed'
    REF_STYLE = lambda i : (2.75 * i, (2.5, 6))
    LIMITS_WIDTH = 2.5
    REF_WIDTH = 1.0

    # Linear states

    ax = plt.subplot(231)
    if ref is not None:
        ax.plot(t[:-1], ref[6, :-1], color=X_REF_COLOR, linestyle=REF_STYLE(0), linewidth=REF_WIDTH)
        ax.plot(t[:-1], ref[7, :-1], color=Y_REF_COLOR, linestyle=REF_STYLE(1), linewidth=REF_WIDTH)
        ax.plot(t[:-1], ref[8, :-1], color=Z_REF_COLOR, linestyle=REF_STYLE(2), linewidth=REF_WIDTH)
    ax.plot(t[:-1], x[6, :-1], color=X_COLOR, linestyle=PRIMARY_STYLE, label=r'$v_x$')
    ax.plot(t[:-1], x[7, :-1], color=Y_COLOR, linestyle=SECONDARY_STYLE, label=r'$v_y$')
    ax.plot(t[:-1], x[8, :-1], color=Z_COLOR, label=r'$v_z$')
    ax.set_xticklabels([])
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
        ax.plot(t[:-1], np.rad2deg(ref[0, :-1]), color=Y_REF_COLOR, linestyle=REF_STYLE(0), linewidth=REF_WIDTH)
        ax.plot(t[:-1], np.rad2deg(ref[1, :-1]), color=X_REF_COLOR, linestyle=REF_STYLE(1), linewidth=REF_WIDTH)
        ax.plot(t[:-1], np.rad2deg(ref[2, :-1]), color=ROLL_REF_COLOR, linestyle=REF_STYLE(2), linewidth=REF_WIDTH)
    ax.plot(t[:-1], np.rad2deg(x[0, :-1]), color=Y_COLOR, label=r'$\omega_x$')
    ax.plot(t[:-1], np.rad2deg(x[1, :-1]), color=X_COLOR, label=r'$\omega_y$')
    ax.plot(t[:-1], np.rad2deg(x[2, :-1]), color=ROLL_COLOR, label=r'$\omega_z$')
    ax.set_xticklabels([])
    ax.set_ylabel(r'$\boldsymbol{\omega} \quad (\frac{^\circ}{\text{s}})$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    ax = plt.subplot(235)
    ax.axhline(10, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.axhline(-10, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    if ref is not None:
        ax.plot(t[:-1], np.rad2deg(ref[3, :-1]), color=Y_REF_COLOR, linestyle=REF_STYLE(0), linewidth=REF_WIDTH)
        ax.plot(t[:-1], np.rad2deg(ref[4, :-1]), color=X_REF_COLOR, linestyle=REF_STYLE(1), linewidth=REF_WIDTH)
        ax.plot(t[:-1], np.rad2deg(ref[5, :-1]), color=ROLL_REF_COLOR, linestyle=REF_STYLE(2), linewidth=REF_WIDTH)
    ax.plot(t[:-1], np.rad2deg(x[3, :-1]), color=Y_COLOR, label=r'$\alpha$')
    ax.plot(t[:-1], np.rad2deg(x[4, :-1]), color=X_COLOR, label=r'$\beta$')
    ax.plot(t[:-1], np.rad2deg(x[5, :-1]), color=ROLL_COLOR, label=r'$\gamma$')
    ax.set_xlabel(r'$t \quad (s)$')
    ax.set_ylabel(r'$\boldsymbol{\varphi} \quad (^\circ)$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    # Inputs

    ax = plt.subplot(233)
    ax.axhline(15, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.axhline(-15, color=LIMITS_COLOR, linestyle=LIMITS_STYLE, linewidth=LIMITS_WIDTH)
    ax.plot(t[:-1], np.rad2deg(u[0, :]), color=Y_COLOR, linestyle='solid', label=r'$\delta_1$')
    ax.plot(t[:-1], np.rad2deg(u[1, :]), color=X_COLOR, linestyle='solid', label=r'$\delta_2$')
    ax.set_xticklabels([])
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
    ax.plot(t[:-1], u[2, :], color=Z_COLOR, label=r'$P_\text{avg}$')
    ax.plot(t[:-1], u[3, :], color=ROLL_COLOR, label=r'$P_\text{diff}$')
    ax.set_yticks(np.arange(-20, 80 + 1, 20))
    ax.set_xlabel(r'$t \quad (s)$')
    ax.set_ylabel(r'$\boldsymbol{P} \quad (\%)$')
    ax.grid()
    ax.legend()
    enforce_min_yrange(ax)

    
    fig.legend(loc='upper center', ncols=4, handles=[
        Line2D([], [], marker='o', linestyle='', color=X_COLOR, label='$x$ subsystem'),
        Line2D([], [], marker='o', linestyle='', color=Y_COLOR, label='$y$ subsystem'),
        Line2D([], [], marker='o', linestyle='', color=Z_COLOR, label='$z$ subsystem'),
        Line2D([], [], marker='o', linestyle='', color=ROLL_COLOR, label='roll subsystem')
    ])
    fig.subplots_adjust(top=0.92)

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.92])
    save_fig(name)
