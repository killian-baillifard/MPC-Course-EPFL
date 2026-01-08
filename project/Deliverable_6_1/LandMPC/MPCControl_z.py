import numpy as np
import cvxpy as cp
from cvxpy import Expression, Constraint
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])   # [vz, z]
    u_ids: np.ndarray = np.array([2])       # thrust

    def __init__(self, A, B, xs, us, Ts, H):
        self.Q  = np.diag([4.0, 40.0])   # [vz, z]
        self.R  = np.diag([0.4])
        self.Qf = self.Q.copy()

        self.xref_default = np.array([0.0, 3.0])  # [vz, z]

        self.x_min = np.array([-20.0, 0.0])
        self.x_max = np.array([ 20.0, 50.0])
        self.u_min = np.array([0.0])
        self.u_max = np.array([100.0])

        super().__init__(A, B, xs, us, Ts, H)

        self.uref_default = self.us.copy()

        self._xbar = None

    def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
       return self.Q, self.R

    @staticmethod
    def _maximal_invariant_set(Acl: np.ndarray, X: Polyhedron, U: Polyhedron, K: np.ndarray, iters: int = 100) -> Polyhedron:  
        Xf = X  
        for _ in range(iters): 
            pre = Polyhedron.from_Hrep(Xf.A @ Acl, Xf.b)
            ku  = Polyhedron.from_Hrep(U.A @ K,    U.b)  
            Xnew = Xf.intersect(pre).intersect(ku).intersect(X)
            Xnew.minHrep(True)  
            Xf.minHrep(True) 
            if Xnew == Xf:  
                Xf = Xnew  
                break  
            Xf = Xnew  
        return Xf


    def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:
        Q, R = self._get_stage_cost()
        K_lqr, P, _ = dlqr(self.A, self.B, Q, R)
        self.K = -K_lqr
        Acl = self.A + self.B @ self.K

        # bounds
        vz_min, vz_max = float(self.x_min[0]), float(self.x_max[0]) 
        z_min, z_max = float(self.x_min[1]), float(self.x_max[1]) 
        u_min, u_max = float(self.u_min[0]), float(self.u_max[0])

        # disturbance bounds
        w_min, w_max = -15.0, 5.0
        w_abs =  float(max(abs(w_min), abs(w_max)))

        # disturbance channel for [vz, z]
        Bd_w = self.B

        # box over-approx invariant set E
        M = 80
        A_pow = np.eye(self.nx)
        e_bound = np.zeros((self.nx, 1))
        for _ in range(M):
            e_bound += np.abs(A_pow @ Bd_w) * w_abs
            A_pow = A_pow @ Acl

        evz = float(e_bound[0, 0])
        ez  = float(e_bound[1, 0])

        E = Polyhedron.from_Hrep(
            np.array([[ 1.0, 0.0],
                      [-1.0, 0.0],
                      [ 0.0, 1.0],
                      [ 0.0,-1.0]]),
            np.array([evz, evz, ez, ez])
        )
        E.minHrep(True)
        self.E = E

        # original constraints X, U
        X = Polyhedron.from_Hrep(
            np.array([[ 1.0, 0.0],
                      [-1.0, 0.0],
                      [ 0.0, 1.0],
                      [ 0.0,-1.0]]),
            np.array([vz_max, -vz_min, z_max, -z_min])
        )
        X.minHrep(True)

        U = Polyhedron.from_Hrep(
            np.array([[ 1.0],
                      [-1.0]]),
            np.array([u_max, -u_min])
        )
        U.minHrep(True)

        # tightened constraints
        Xbar = X - E
        Xbar.minHrep(True)
        self.Xbar = Xbar

        KE = self.K @ E 
        Ubar = U - KE
        Ubar.minHrep(True)
        self.Ubar = Ubar

        Xf = self._maximal_invariant_set(Acl, Xbar, Ubar, self.K, iters=200)
        Xf.minHrep(True)
        self.Xf = Xf

        constraints: list[cp.Constraint] = []

        constraints += [Xbar.A @ self.x_var[:, 0:self.N] <= Xbar.b.reshape(-1, 1)]  
        constraints += [Xbar.A @ self.x_var[:, self.N] <= Xbar.b.reshape(-1,)] 
        constraints += [Ubar.A @ self.u_var[:, 0:self.N] <= Ubar.b.reshape(-1, 1)] 

        constraints += [Xf.A @ self.x_var[:, self.N] <= Xf.b.reshape(-1,)] 

        terminal_cost = cp.quad_form(self.x_var[:, self.N] - self._xref_par, P)  
        return terminal_cost, constraints

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):

        x_real = x0.copy() if x0.shape[0] == self.nx else x0[self.x_ids].copy()

        xref = self.xref_default.copy() if x_target is None else (x_target.copy() if x_target.shape[0] == self.nx else x_target[self.x_ids].copy())  
        uref = self.uref_default.copy() if u_target is None else (u_target.copy() if u_target.shape[0] == self.nu else u_target[self.u_ids].copy())
        
        if self._xbar is None:
            self._xbar = x_real.copy()
        xbar0 = self._xbar.copy()

        self._x0_par.value = xbar0
        self._xref_par.value = xref
        self._uref_par.value = uref

        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        if self.u_var.value is None:
            u0 = self.us.copy()
            x_traj = np.tile(xbar0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        x_traj = self.x_var.value
        u_traj = self.u_var.value

        e0 = x_real - xbar0
        u0 = u_traj[:, 0] + (self.K @ e0.reshape(self.nx, 1)).reshape(self.nu,)

        self._xbar = x_traj[:, 1].copy()

        return u0, x_traj, u_traj


