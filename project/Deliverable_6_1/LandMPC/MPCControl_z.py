import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        # weights
        Q = np.diag([40.0, 4.0])
        R = np.diag([0.4])

        # bounds (keep as constants here; adjust to your provided limits if different)
        z_min, z_max = 0.0, 50.0
        vz_min, vz_max = -20.0, 20.0
        u_min, u_max = 0.0, 30.0

        # LQR feedback for tube
        K_lqr, P, _ = dlqr(self.A, self.B, Q, R)
        self.K = -K_lqr
        Acl = self.A + self.B @ self.K

        # disturbance channel for z-subsystem (acceleration mismatch)
        Ts = float(self.Ts)
        Bd_w = np.array([[0.5 * Ts * Ts], [Ts]])

        # bounded disturbance
        w_min, w_max = -15.0, 5.0
        W = Polyhedron.from_box(np.array([[w_min], [w_max]]))

        # approximate mRPI E
        M = 20
        E = Polyhedron.from_Hrep(np.zeros((1, self.nx)), np.zeros((1,)))
        A_pow = np.eye(self.nx)
        for _ in range(M):
            E = E + (A_pow @ Bd_w) @ W
            A_pow = A_pow @ Acl
        E.minHrep(True)
        _ = E.Vrep
        self.E = E

        # original constraints as polyhedra
        X = Polyhedron.from_box(np.array([[z_min, vz_min], [z_max, vz_max]]))
        U = Polyhedron.from_box(np.array([[u_min], [u_max]]))

        # tightened constraints
        Xbar = X - E
        Ubar = U - (self.K @ E)
        Xbar.minHrep(True)
        Ubar.minHrep(True)
        _ = Xbar.Vrep
        _ = Ubar.Vrep
        self.Xbar = Xbar
        self.Ubar = Ubar

        # optimization variables (use template names)
        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))

        # parameters (use template names)
        x0_par = cp.Parameter(self.nx)
        xref_par = cp.Parameter(self.nx)
        uref_par = cp.Parameter(self.nu)

        self._x0_par = x0_par
        self._xref_par = xref_par
        self._uref_par = uref_par

        dx = x_var - cp.reshape(xref_par, (self.nx, 1))
        du = u_var - cp.reshape(uref_par, (self.nu, 1))

        cost = 0
        constraints = []

        constraints += [x_var[:, 0] == x0_par]

        for k in range(self.N):
            constraints += [x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k]]
            constraints += [self.Xbar.A @ x_var[:, k] <= self.Xbar.b]
            constraints += [self.Ubar.A @ u_var[:, k] <= self.Ubar.b]
            cost += cp.quad_form(dx[:, k], Q)
            cost += cp.quad_form(du[:, k], R)

        constraints += [self.Xbar.A @ x_var[:, self.N] <= self.Xbar.b]
        cost += cp.quad_form(dx[:, self.N], P)

        self._x_var = x_var
        self._u_var = u_var

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0_red = x0[self.x_ids]

        xref = self.xs if x_target is None else x_target[self.x_ids]
        uref = self.us if u_target is None else u_target[self.u_ids]

        # nominal initial state: use measured state (simple and consistent with template)
        self._x0_par.value = x0_red
        self._xref_par.value = xref
        self._uref_par.value = uref

        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        if self._u_var.value is None:
            u0 = self.us.copy()
            x_traj = np.tile(x0_red.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        # nominal predicted trajectories
        x_traj = self._x_var.value
        u_traj = self._u_var.value

        # tube law (use nominal first input + feedback on current nominal error)
        e0 = x0_red - x_traj[:, 0]
        u0 = u_traj[:, 0] + (self.K @ e0.reshape(self.nx, 1)).reshape(self.nu,)

        return u0, x_traj, u_traj

