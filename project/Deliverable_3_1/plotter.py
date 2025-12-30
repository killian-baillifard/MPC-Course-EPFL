from matplotlib import pyplot as plt
import numpy as np
from LinearMPC.MPCVelControl import MPCVelControl
import os

def start_fig(name: str) -> None:
    FIGS_SIZES = (12, 5)
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
    plt.savefig(PLOT_DIR + file_name(name))
    plt.show()

def plot_x_y_terminal_set(name: str, mpc: MPCVelControl) -> None:

    start_fig(name)
    OPACITY = 0.1
    X_COLOR = 'teal'
    Y_COLOR = 'orange'

    # Plot first dimensions pair

    ax = plt.subplot(221)
    mpc.mpc_x.O_inf.projection((1, 0)).plot(ax, color=X_COLOR, opacity=OPACITY)
    plt.ylabel(r'$\omega_\beta \quad (rad/s)$')
    plt.grid()

    ax = plt.subplot(222)
    mpc.mpc_y.O_inf.projection((1, 0)).plot(ax, color=Y_COLOR, opacity=OPACITY)
    plt.ylabel(r'$\omega_\alpha \quad (rad/s)$')
    plt.grid()

    # Plot second dimensions pair

    ax = plt.subplot(223)
    mpc.mpc_x.O_inf.projection((1, 2)).plot(ax, color=X_COLOR, opacity=OPACITY)
    plt.xlabel(r'$\beta \quad (rad)$')
    plt.ylabel(r'$v_x \quad (m/s)$')
    plt.grid()

    ax = plt.subplot(224)
    mpc.mpc_y.O_inf.projection((1, 2)).plot(ax, color=Y_COLOR, opacity=OPACITY)
    plt.xlabel(r'$\alpha \quad (rad)$')
    plt.ylabel(r'$v_y \quad (m/s)$')
    plt.grid()

    save_fig(name)

def plot_traj(name: str, x: np.ndarray, u: np.ndarray, t: np.ndarray) -> None:

    start_fig(name)
    X_COLOR = 'xkcd:red'
    Y_COLOR = 'xkcd:green'
    Z_COLOR = 'xkcd:blue'

    plt.subplot(221)
    plt.plot(t[:-1], np.rad2deg(x[0, :-1]), color=X_COLOR, label=r'$\omega_\alpha$')
    plt.plot(t[:-1], np.rad2deg(x[1, :-1]), color=Y_COLOR, label=r'$\omega_\beta$')
    plt.plot(t[:-1], np.rad2deg(x[2, :-1]), color=Z_COLOR, label=r'$\omega_\gamma$')
    plt.ylabel(r'$\omega_x, \quad \omega_y, \quad \omega_z \quad (^\circ/s)$')
    plt.grid()
    plt.legend(loc='upper right')

    plt.subplot(222)
    plt.plot(t[:-1], np.rad2deg(x[3, :-1]), color=X_COLOR, label=r'$\alpha$')
    plt.plot(t[:-1], np.rad2deg(x[4, :-1]), color=Y_COLOR, label=r'$\beta$')
    plt.plot(t[:-1], np.rad2deg(x[5, :-1]), color=Z_COLOR, label=r'$\gamma$')
    plt.ylabel(r'$\alpha, \quad \beta, \quad \gamma \quad (^\circ)$')
    plt.grid()
    plt.legend(loc='upper right')

    plt.subplot(223)
    plt.plot(t[:-1], x[6, :-1], color=X_COLOR, linestyle=(0, (1, 3)), label=r'$v_x$')
    plt.plot(t[:-1], x[7, :-1], color=Y_COLOR, linestyle=(2, (1, 3)), label=r'$v_y$')
    plt.plot(t[:-1], x[8, :-1], color=Z_COLOR, label=r'$v_z$')
    plt.xlabel(r'$t \quad (s)$')
    plt.ylabel(r'$v_x, \quad v_y, \quad v_z \quad (m/s)$')
    plt.grid()
    plt.legend(loc='upper right')

    ax = plt.subplot(224)
    twin = ax.twinx()
    ax.plot(t[:-1], np.rad2deg(u[0, :]), color=X_COLOR, label=r'$\delta_1$')
    ax.plot(t[:-1], np.rad2deg(u[1, :]), color=Y_COLOR, label=r'$\delta_2$')
    twin.plot(t[:-1], u[2, :], color='xkcd:teal', label=r'$P_\text{avg}$')
    twin.plot(t[:-1], u[3, :], color='xkcd:blue', label=r'$P_\text{diff}$')
    ax.set_xlabel(r'$t \quad (s)$')
    ax.set_ylabel(r'$\delta_1, \quad \delta_2 \quad (^\circ)$')
    twin.set_ylabel(r'$P_\text{avg}, \quad P_\text{diff} \quad (\%)$')
    ax.legend(loc='upper right')
    twin.grid()
    twin.legend(loc='lower right')

    save_fig(name)
