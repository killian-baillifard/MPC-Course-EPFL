from .MPCControl_base import MPCControl_base
import numpy as np
from control import dlqr
import cvxpy as cp
from cvxpy import Expression, Constraint

BETA_MAX 		= 10.0					# °
DELTA_2_MAX 	= 15.0					# °

OMEGA_BETA_TYP 	= 10.0					# °/s
BETA_TYP 		= BETA_MAX / 2.0		# °
V_X_TYP 		= 1.0					# m/s
DELTA_2_TYP 	= DELTA_2_MAX / 2.0		# °

class MPCControl_xvel(MPCControl_base):

	x_ids = np.array([1, 4, 6])
	u_ids = np.array([1])

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.diag([
			1.0 / np.deg2rad(OMEGA_BETA_TYP)**2,	# omega_beta cost
			1.0 / np.deg2rad(BETA_TYP)**2,			# beta cost
			1.0 / V_X_TYP**2						# v_x cost
		])
		R = np.diag([
			1.0 / np.deg2rad(DELTA_2_TYP)**2		# delta_2 cost
		])
		return Q, R

	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:

		# Define terminal cost

		Q, _ = self._get_stage_cost()
		terminalCost = cp.quad_form(self.dx_var[:, -1], Q)

		# Define constraints with slack variable

		self.epsilon_var = cp.Variable((1, self.N + 1), 'epsilon', nonneg=True)
		constraints = [
			self.x_var[1] 	<= +BETA_MAX + self.epsilon_var[0],
			self.x_var[1] 	>= -BETA_MAX - self.epsilon_var[0],
			self.u_var		<= +np.deg2rad(DELTA_2_MAX),
			self.u_var 		>= -np.deg2rad(DELTA_2_MAX)
		]

		# Add slack cost

		S = 10.0 / np.deg2rad(BETA_MAX)**2
		for i in range(self.N + 1):
			terminalCost += S * cp.norm1(self.epsilon_var[:, i])

		# Return cost and constraints
		
		return terminalCost, constraints
