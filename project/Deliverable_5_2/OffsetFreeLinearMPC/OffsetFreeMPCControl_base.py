import numpy as np
import cvxpy as cp
from abc import abstractmethod
from .MPCControl_base import MPCControl_base

class OffsetFreeMPCControl_base(MPCControl_base):

	x_ids: np.ndarray
	u_ids: np.ndarray

	def __init__(
		self,
		A: np.ndarray,
		B: np.ndarray,
		xs: np.ndarray,
		us: np.ndarray,
		Ts: float,
		H: float,
	) -> None:
		
		# Save controller configuration

		self.N = int(H / Ts)
		self.NX = self.x_ids.shape[0]
		self.NU = self.u_ids.shape[0]
		self.Ts = Ts
		self.xs = xs[self.x_ids].reshape(self.NX, 1)
		self.us = us[self.u_ids].reshape(self.NU, 1)

		# Extract subset of discretized states and inputs

		subA = A[np.meshgrid(self.x_ids, self.x_ids)].T
		subB = B[np.meshgrid(self.x_ids, self.u_ids)].T
		self.A, self.B = self._discretize(subA, subB, Ts)

		# Create optimization variables and parameters with delta formulation

		self.xs_cst	 = cp.Constant(self.xs, name='xs')
		self.x0_par	 = cp.Parameter((self.NX, 1), name='x0')
		self.x_var 	 = cp.Variable((self.NX, self.N + 1), name='x')
		self.dx_var	 = cp.Variable((self.NX, self.N + 1), name='dx')
		self.xt_par  = cp.Parameter((self.NX, 1), name='xt')

		self.us_cst	= cp.Constant(self.us, name='us')
		self.u_var 	= cp.Variable((self.NU, self.N), name='u')
		self.du_var	= cp.Variable((self.NU, self.N), name='du')

		self.d_par	= cp.Parameter((self.NU, 1), name='d_par')

		# Define trajectory cost

		Q, R = self._get_stage_cost()
		cost = 0
		for i in range(self.N):
			cost += cp.quad_form(self.dx_var[:, i], Q)
			cost += cp.quad_form(self.du_var[:, i], R)

		# Initialize estimator

		Bd = self._setup_estimator()
		self.u_prev = np.zeros((self.NU, 1))

		# Define delta formulation with disturbance

		dynamics = [
			self.dx_var			== self.x_var - self.xs_cst - self.xt_par,
			self.du_var			== self.u_var - self.us_cst,
			self.dx_var[:, 0] 	== self.x0_par[:, 0] - self.xs_cst[:, 0] - self.xt_par[:, 0],
			self.dx_var[:, 1:] 	== self.A @ self.dx_var[:, :-1] + self.B @ self.du_var + Bd @ self.d_par,
		]

		# Create optimization problem

		terminalCost, constraints = self._get_terminal_cost_and_constraints()
		self.ocp = cp.Problem(cp.Minimize(cost + terminalCost), dynamics + constraints)

	@abstractmethod
	def _setup_estimator(self) -> np.ndarray:
		pass

	@abstractmethod
	def _update_estimator(self, x0: np.ndarray, u_last: np.ndarray) -> np.ndarray:
		pass

	def get_u(
		self,
		x0: np.ndarray,
		x_target: np.ndarray = None,
        u_target: np.ndarray = None
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

		# Estimate disturbance

		z_hat = self._update_estimator(x0, self.u_prev)
		self.x0_par.value = z_hat[:self.NX, :]
		self.d_par.value = z_hat[self.NX:, :]
		
		# Solve optimization problem

		self.xt_par.value = x_target.reshape(self.NX, 1)
		self.ocp.solve(solver=cp.PIQP)
		assert self.ocp.status == cp.OPTIMAL

		# Save trajectory

		x_traj = self.x_var.value
		u_traj = self.u_var.value
		u0 = u_traj[:, 0:1]
		self.u_prev = u0

		# Return open loop prediction

		return u0[:,0], x_traj, u_traj
