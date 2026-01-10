from .MPCControl_base import MPCControl_base
import numpy as np
import cvxpy as cp
from cvxpy import Expression, Constraint

P_AVG_MIN 		= 40.0	# %
P_AVG_MAX 		= 80.0	# %

V_Z_TYP 		= 1.0	# m/s
DP_AVG_TYP 		= 10.0	# %

LAMBDA 	= 0.0

class MPCControl_zvel(MPCControl_base):

	x_ids = np.array([8])
	u_ids = np.array([2])

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.exp(LAMBDA) * np.diag([
			1.0 / V_Z_TYP**2		# v_z cost
		])
		R = np.exp(-LAMBDA) * np.diag([
			1.0 / DP_AVG_TYP**2		# P_avg cost
		])
		return Q, R

	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:

		# Define terminal cost
		Q, _ = self._get_stage_cost()
		terminalCost = cp.quad_form(self.dx_var[:, -1], Q)

		# Define constraints
		constraints = [
			self.u_var	>= P_AVG_MIN,	# P_avg >= 40% 
			self.u_var	<= P_AVG_MAX	# P_avg <= 80%
		]
	
		# Return cost and constraints
		return terminalCost, constraints
	
	def setup_estimator(self):
        
		bd = float(self.B[0, 0])

		self.A_aug = np.array([
			[self.A[0, 0], bd],
			[0.0,          1.0]
		])
		self.B_aug = np.array([
			[self.B[0, 0]],
			[0.0]
		])
		self.C_aug = np.array([[1.0, 0.0]])

		Qe = np.diag([1e-2, 1e-2])
		Re = np.array([[1e-2]])
		P = np.eye(2)

		S = self.C_aug @ P @ self.C_aug.T + Re
		K = P @ self.C_aug.T @ np.linalg.inv(S)
		P = self.A_aug @ (P - K @ self.C_aug @ P) @ self.A_aug.T + Qe

		self.K = K
		self.xd_hat = np.zeros((2, 1))
		self.d_estimate = 0.0
		self.x_estimate = 0.0

	def update_estimator(self, x_data, u_data) -> None:

		y = float(np.array(x_data).reshape(-1)[0])
		u = float(np.array(u_data).reshape(-1)[0])

		y_hat = float(self.C_aug @ self.xd_hat)
		self.xd_hat = self.A_aug @ self.xd_hat + self.B_aug * u + self.K * (y - y_hat)

		self.x_estimate = float(self.xd_hat[0, 0])
		self.d_estimate = float(self.xd_hat[1, 0])
