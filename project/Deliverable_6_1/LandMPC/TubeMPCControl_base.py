import numpy as np
import cvxpy as cp
from abc import abstractmethod
from cvxpy import Expression, Constraint
from .MPCControl_base import MPCControl_base

class TubeMPCControl_base(MPCControl_base):

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
		self.xs = xs[self.x_ids]
		self.us = us[self.u_ids]

		# Extract subsystem (continuous) and discretize

		subA = A[np.meshgrid(self.x_ids, self.x_ids)].T
		subB = B[np.meshgrid(self.x_ids, self.u_ids)].T
		self.A, self.B = self._discretize(subA, subB, Ts)

		# Create optimization variables and parameters with delta formulation

		self.xs_cst = cp.Constant(self.xs.reshape(self.NX, 1), name="xs")
		self.dx_var = cp.Variable((self.NX, self.N + 1), name="dz")
		self.x0_par = cp.Parameter((self.NX, 1), name="x0")
		self.xt_par = cp.Parameter((self.NX, 1), name="xt")

		self.us_cst = cp.Constant(self.us.reshape(self.NU, 1), name="us")
		self.du_var = cp.Variable((self.NU, self.N), name="dv")

		# Define trajectory cost

		Q, R = self._get_stage_cost()
		cost = 0
		for k in range(self.N):
			cost += cp.quad_form(self.dx_var[:, k], Q)
			cost += cp.quad_form(self.du_var[:, k], R)

		# Define delta dynamics
		
		dynamics = [
			self.dx_var[:, 1:] == self.A @ self.dx_var[:, :-1] + self.B @ self.du_var
		]

		# Create optimization problem
		
		terminalCost, constraints = self._get_terminal_cost_and_constraints()
		self.ocp = cp.Problem(cp.Minimize(cost + terminalCost), dynamics + constraints)

	@abstractmethod
	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		pass

	@abstractmethod
	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:
		pass

	def get_u(
		self,
		x0: np.ndarray,
		x_target: np.ndarray = None,
		u_target: np.ndarray = None
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		
		# Set default target at origin

		if x_target is None:
			x_target = np.zeros(self.NX)

		# Solve optimization problem

		self.x0_par.value = x0.reshape(self.NX, 1)
		self.xt_par.value = x_target.reshape(self.NX, 1)
		self.ocp.solve(solver=cp.PIQP)
		assert self.ocp.status == cp.OPTIMAL

		# Extract first nominal step
		dz0 = self.dx_var.value[:, 0]
		dv0 = self.du_var.value[:, 0]
		dx0 = (x0.reshape(-1) - self.xs.reshape(-1) - x_target.reshape(-1))
		du0 = dv0 + (self.K @ (dx0 - dz0)).reshape(-1)
		u0 = (self.us.reshape(-1) + du0).reshape(-1)

		# Return open loop prediction

		x_traj = self.dx_var.value + self.xs.reshape(-1, 1) + x_target.reshape(-1, 1)
		u_traj = self.du_var.value + self.us.reshape(-1, 1)
		return u0, x_traj, u_traj
