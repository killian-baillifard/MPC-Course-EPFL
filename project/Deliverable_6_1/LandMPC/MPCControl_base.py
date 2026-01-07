import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))

        x0_par = cp.Parameter(self.nx)
        xref_par = cp.Parameter(self.nx)
        uref_par = cp.Parameter(self.nu)

        self._x0_par = x0_par
        self._xref_par = xref_par
        self._uref_par = uref_par

        self.x_var = x
        self.u_var = u

        Q, R = self._get_stage_cost()
        Qf = self.Qf if hasattr(self, "Qf") else Q

        cost = 0
        constraints: list[cp.Constraint] = []

        constraints += [x[:, 0] == x0_par]

        dx = x - cp.reshape(xref_par, (self.nx, 1))
        du = u - cp.reshape(uref_par, (self.nu, 1))

        for k in range(self.N):
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]
            cost += cp.quad_form(dx[:, k], Q)
            cost += cp.quad_form(du[:, k], R)

            if hasattr(self, "x_min"):
                constraints += [x[:, k] >= self.x_min]
                constraints += [x[:, k] <= self.x_max]

            if hasattr(self, "u_min"):
                constraints += [u[:, k] >= self.u_min]
                constraints += [u[:, k] <= self.u_max]

        cost += cp.quad_form(dx[:, self.N], Qf)

        if hasattr(self, "x_min"):
            constraints += [x[:, self.N] >= self.x_min]
            constraints += [x[:, self.N] <= self.x_max]

        terminal_cost, terminal_constraints = self._get_terminal_cost_and_constraints()
        cost += terminal_cost
        constraints += terminal_constraints
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), dt=Ts)
        return Ad, Bd

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if x0.shape[0] != self.nx:  
            x0_red = x0[self.x_ids].copy()  
        else:  
            x0_red = x0.copy()  

        if x_target is None: 
            xref = self.xs.copy()  
        else:  
            xref = x_target[self.x_ids].copy() if x_target.shape[0] != self.nx else x_target.copy()  

        if u_target is None: 
            uref = self.us.copy()  
        else: 
            uref = u_target[self.u_ids].copy() if u_target.shape[0] != self.nu else u_target.copy()  

        self._x0_par.value = x0_red
        self._xref_par.value = xref
        self._uref_par.value = uref
        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        if self._u_var.value is None:
            u0 = self.us
            x_traj = np.tile(self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj
        
        u0 = self._u_var.value[:, 0]
        x_traj = self._x_var.value
        u_traj = self._u_var.value

        return u0, x_traj, u_traj
    
    def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:  
        return self.Q, self.R  

    def _get_terminal_cost_and_constraints(self) -> tuple[cp.Expression, list[cp.Constraint]]:  
        terminal_cost = 0.0  
        terminal_constraints: list[cp.Constraint] = []  
        return terminal_cost, terminal_constraints  