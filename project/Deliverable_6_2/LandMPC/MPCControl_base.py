import numpy as np
import cvxpy as cp
from abc import abstractmethod
from cvxpy import Expression, Constraint
from mpt4py import Polyhedron
from scipy.signal import cont2discrete

class MPCControl_base:

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

		# Extract subset of discretized states and inputs

		subA = A[np.meshgrid(self.x_ids, self.x_ids)].T
		subB = B[np.meshgrid(self.x_ids, self.u_ids)].T
		self.A, self.B = self._discretize(subA, subB, Ts)

		# Create optimization variables and parameters with delta formulation

		self.xs_cst	 = cp.Constant(self.xs.reshape(self.NX, 1), name='xs')
		self.x_var 	 = cp.Variable((self.NX, self.N + 1), name='x')
		self.dx_var	 = cp.Variable((self.NX, self.N + 1), name='dx')
		self.x0_par	 = cp.Parameter((self.NX, 1), name='x0')

		self.us_cst	= cp.Constant(self.us.reshape(self.NU, 1), name='us')
		self.u_var 	= cp.Variable((self.NU, self.N), name='u')
		self.du_var	= cp.Variable((self.NU, self.N), name='du')

		# Define trajectory cost

		Q, R = self._get_stage_cost()
		cost = 0
		for i in range(self.N):
			cost += cp.quad_form(self.dx_var[:, i], Q)
			cost += cp.quad_form(self.du_var[:, i], R)

		# Define delta formulation (nominal MPC uses equality at k=0)

		dynamics = [
			self.dx_var			== self.x_var - self.xs_cst,
			self.du_var			== self.u_var - self.us_cst,
			self.dx_var[:, 0] 	== self.x0_par[:, 0] - self.xs_cst[:, 0],
			self.dx_var[:, 1:] 	== self.A @ self.dx_var[:, :-1] + self.B @ self.du_var,
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

	@staticmethod
	def _max_invariant_set(O: Polyhedron, A_cl: np.ndarray, max_iter: int = 30) -> Polyhedron:
		for _ in range(max_iter):
			Oprev = O
			O = Polyhedron.from_Hrep(np.vstack([O.A, O.A @ A_cl]), np.vstack([O.b, O.b]).reshape(-1))
			O.minHrep(True)
			_ = O.Vrep
			if O == Oprev:
				return O
		raise RuntimeError('Did not converge to maximum invariant set')

	@staticmethod
	def _min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 30) -> Polyhedron:
		nx = A_cl.shape[0]
		Oprev = W
		A_cl_ith_power = np.eye(nx)
		for itr in range(max_iter):
			A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
			O = Oprev + A_cl_ith_power @ W
			O.minHrep(True)
			if np.linalg.matrix_norm(A_cl_ith_power, ord=2) < 1e-2:
				return O
			Oprev = O
		raise RuntimeError('Did not converge to maximum invariant set')

	@staticmethod
	def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
		NX, NU = B.shape
		C = np.zeros((1, NX))
		D = np.zeros((1, NU))
		A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
		return A_discrete, B_discrete

	def get_u(
		self,
		x0: np.ndarray,
		x_target: np.ndarray = None,
        u_target: np.ndarray = None
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		
		# Solve optimization problem

		self.x0_par.value = x0.reshape(self.NX, 1)
		self.ocp.solve(solver=cp.PIQP)
		assert self.ocp.status == cp.OPTIMAL

		# Return open loop prediction

		x_traj = self.x_var.value
		u_traj = self.u_var.value
		u0 = u_traj[:, 0]
		return u0, x_traj, u_traj
