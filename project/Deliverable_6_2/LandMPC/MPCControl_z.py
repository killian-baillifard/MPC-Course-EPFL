from .TubeMPCControl_base import TubeMPCControl_base
import numpy as np
from control import dlqr
import cvxpy as cp
from mpt4py import Polyhedron
from cvxpy import Expression, Constraint

P_AVG_MIN 		= 40.0	# %
P_AVG_MAX 		= 80.0	# %

V_Z_TYP 		= 1.3	# m/s
P_Z_TYP			= 0.5	# m
DP_AVG_TYP 		= 17.0	# %

W_MIN = -15.0
W_MAX = 5.0

class MPCControl_z(TubeMPCControl_base):

	x_ids: np.ndarray = np.array([8, 11])
	u_ids: np.ndarray = np.array([2])

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.diag([
			1.0 / V_Z_TYP**2,		# v_z cost
			1.0 / P_Z_TYP**2		# p_z cost
		])
		R = np.diag([
			1.0 / DP_AVG_TYP**2		# delta P_avg cost
		])
		return Q, R

	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:

		# Compute terminal controller

		Q, R = self._get_stage_cost()
		K, Qf, _ = dlqr(self.A, self.B, Q, R)
		K = -K
		self.K = np.array(K)
		terminalCost = cp.quad_form(self.dz_var[:, -1], Qf)

		# Define offset state constraint set

		z_s = float(self.xs.reshape(-1)[1])
		F = np.array([
			[0.0, -1.0],   	# dp_z >= 0
		])
		f = np.array([
			z_s				# dp_z >= 0
		])
		X = Polyhedron.from_Hrep(F, f)

		# Define offset input constraints set

		us = float(self.us.reshape(-1)[0])
		du_min = P_AVG_MIN - us
		du_max = P_AVG_MAX - us
		G = np.array([
			[-1.0],			# dP_avg >= 40%
			[+1.0],			# dP_avg <= 80%
		])
		g = np.array([
			-du_min,		# dP_avg >= 40%
			+du_max,		# dP_avg <= 80%
		])
		U = Polyhedron.from_Hrep(G, g)

		# Define disturbance set

		H = np.array([
			[+1.0],   		# w <= W_MAX
			[-1.0],   		# w >= W_MIN
		])
		h = np.array([
			W_MAX,			# w <= W_MAX
			-W_MIN,			# w >= W_MIN
		])
		W = Polyhedron.from_Hrep(H, h)

		# Compute minimal robust invariant set for error dynamics

		A_cl = self.A + self.B @ self.K
		BW = self.B @ W
		E = self._min_robust_invariant_set(A_cl, BW, self.N)
		self.E = E

		# Tighten state and input constraints

		X_tilde = X - E
		U_tilde = U - (self.K @ E)
		self.X_tilde = X_tilde
		self.U_tilde = U_tilde

		# Compute terminal tightened set

		X_and_KU = X_tilde.intersect(Polyhedron.from_Hrep(U_tilde.A @ self.K, U_tilde.b))
		Xf_tilde = self._max_invariant_set(X_and_KU, A_cl, self.N)
		self.Xf_tilde = Xf_tilde

		# Define constraints

		dx0_expr = self.x0_par[:, 0] - self.xs_cst[:, 0]
		constraints = [
			self.E.A @ (dx0_expr - self.dz_var[:, 0]) 	<= self.E.b,
			X_tilde.A @ self.dz_var[:, :-1] 			<= X_tilde.b.reshape(-1, 1),
			U_tilde.A @ self.dv_var 					<= U_tilde.b.reshape(-1, 1),
			Xf_tilde.A @ self.dz_var[:, -1] 			<= Xf_tilde.b.reshape(-1, 1)
		]

		# Return cost and constraints

		return terminalCost, constraints
