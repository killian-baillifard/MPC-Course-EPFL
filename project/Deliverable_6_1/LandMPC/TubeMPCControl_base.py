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

		# Optimization variables: nominal tube center (dz) and nominal input (dv)
		self.dx_var = cp.Variable((self.NX, self.N + 1), name="dz")  # tube centers (delta)
		self.du_var = cp.Variable((self.NU, self.N), name="dv")      # nominal inputs (delta)

		# Parameters
		self.x0_par = cp.Parameter((self.NX, 1), name="x0")          # measured state (subsystem)
		self.xt_par = cp.Parameter((self.NX, 1), name="xt")          # target shift

		# Constants
		self.xs_cst = cp.Constant(self.xs.reshape(self.NX, 1), name="xs")
		self.us_cst = cp.Constant(self.us.reshape(self.NU, 1), name="us")

		# Measured delta state (for tube init)
		dx0_expr = self.x0_par[:, 0] - self.xs_cst[:, 0] - self.xt_par[:, 0]

		# Subclass terminal cost + ALL constraints (tightened constraints etc.)
		terminalCost, constraints = self._get_terminal_cost_and_constraints()

		# Require tube ingredients
		assert hasattr(self, "K"), "TubeMPCControl_base expects self.K to be set in _get_terminal_cost_and_constraints()"
		assert hasattr(self, "E"), "TubeMPCControl_base expects self.E to be set in _get_terminal_cost_and_constraints()"

		# Tube initial inclusion: dx0 - dz0 ∈ E  <=>  E.A (dx0 - dz0) <= E.b
		tube_init = [self.E.A @ (dx0_expr - self.dx_var[:, 0]) <= self.E.b]

		# Nominal dynamics: dz_{k+1} = A dz_k + B dv_k
		dynamics = [self.dx_var[:, 1:] == self.A @ self.dx_var[:, :-1] + self.B @ self.du_var]

		# Stage cost
		Q, R = self._get_stage_cost()
		cost = 0
		for k in range(self.N):
			cost += cp.quad_form(self.dx_var[:, k], Q)
			cost += cp.quad_form(self.du_var[:, k], R)
		cost += terminalCost

		self.ocp = cp.Problem(cp.Minimize(cost), tube_init + dynamics + constraints)

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

		if x_target is None:
			x_target = np.zeros((self.NX,))

		# Set parameters
		self.xt_par.value = x_target.reshape(self.NX, 1)
		self.x0_par.value = x0.reshape(self.NX, 1)

		# Solve
		self.ocp.solve(solver=cp.PIQP)
		assert self.ocp.status == cp.OPTIMAL

		# Extract first nominal step
		dz0 = self.dx_var.value[:, 0]
		dv0 = self.du_var.value[:, 0]

		# Compute measured delta state
		dx0 = (x0.reshape(-1) - self.xs.reshape(-1) - x_target.reshape(-1))

		# Tube law: du = dv + K (dx - dz)
		du0 = dv0 + (self.K @ (dx0 - dz0)).reshape(-1)

		# Absolute control
		u0 = (self.us.reshape(-1) + du0).reshape(-1)

		# Build output trajectories in ABSOLUTE coords
		x_traj = np.zeros((self.NX, self.N + 1))
		u_traj = np.zeros((self.NU, self.N))

		for k in range(self.N + 1):
			x_traj[:, k] = self.dx_var.value[:, k] + self.xs.reshape(-1) + x_target.reshape(-1)

		for k in range(self.N):
			u_traj[:, k] = self.du_var.value[:, k] + self.us.reshape(-1)

		return u0, x_traj, u_traj
