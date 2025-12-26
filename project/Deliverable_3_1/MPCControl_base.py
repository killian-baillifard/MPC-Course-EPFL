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
		self.x_var 			= cp.Variable((self.NX, self.N + 1), name='x')
		self.dx_var			= cp.Variable((self.NX, self.N + 1), name='dx')
		self.xs_par			= cp.Parameter((self.NX, 1), name='xs')

		self.u_var 			= cp.Variable((self.NU, self.N), name='u')
		self.du_var			= cp.Variable((self.NU, self.N), name='du')
		self.us_par			= cp.Parameter((self.NU, 1), name='us')

		self.dx0_par		= cp.Parameter((self.NX, 1), name='dx0')
		self.xs_par.value 	= self.xs.reshape(self.NX, 1)
		self.us_par.value 	= self.us.reshape(self.NU, 1)

		# Create optimization problem
		cost, constraints = self._get_cost_and_constraints()
		self.ocp = cp.Problem(cp.Minimize(cost), constraints)

	@abstractmethod
	def _get_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:
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
		
		# Allocate outputs
		x_traj = np.zeros((self.NX, self.N + 1))
		u_traj = np.zeros((self.NU, self.N))

		# Compute steady state error
		x_traj[:, 0] = x0
		dxk = x0 - self.xs

		# Closed-loop simulation
		for k in range(self.N):

			# Solve step
			self.dx0_par.value = dxk.reshape(self.NX, 1)
			self.ocp.solve(solver=cp.PIQP)
			assert self.ocp.status == cp.OPTIMAL

			# Simulate next state
			duk = self.du_var.value[:, 0]
			dxk = self.A @ dxk + self.B @ duk

			# Save trajectory
			x_traj[:, k + 1] = dxk + self.xs
			u_traj[:, k] = duk + self.us

		# Return predicted input and trajectories
		u0 = u_traj[:, 0]
		return u0, x_traj, u_traj
