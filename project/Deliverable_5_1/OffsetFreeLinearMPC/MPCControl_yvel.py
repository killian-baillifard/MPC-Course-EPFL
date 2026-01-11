from .MPCControl_base import MPCControl_base
import numpy as np
from control import dlqr
import cvxpy as cp
from cvxpy import Expression, Constraint

ALPHA_MAX 		= 10.0					# °
DELTA_1_MAX 	= 15.0					# °

OMEGA_ALPHA_TYP = 10.0					# °/s
ALPHA_TYP 		= ALPHA_MAX / 2.0		# °
V_Y_TYP 		= 1.0					# m/s
DELTA_1_TYP 	= DELTA_1_MAX / 2.0		# °

class MPCControl_yvel(MPCControl_base):

	x_ids = np.array([0, 3, 7])
	u_ids = np.array([0])

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.diag([
			1.0 / np.deg2rad(OMEGA_ALPHA_TYP)**2,	# omega_alpha cost
			1.0 / np.deg2rad(ALPHA_TYP)**2,			# alpha cost
			1.0 / V_Y_TYP**2						# v_y cost
		])
		R = np.diag([
			1.0 / np.deg2rad(DELTA_1_TYP)**2		# delta_1 cost
		])
		return Q, R

	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:
		
		# Define terminal cost

		Q, _ = self._get_stage_cost()
		terminalCost = cp.quad_form(self.dx_var[:, -1], Q)

		# Define constraints with slack variable

		self.epsilon_var = cp.Variable((1, self.N), 'epsilon', nonneg=True)
		constraints = [
			self.x_var[1, :-1] 	<= +ALPHA_MAX + self.epsilon_var[0, :],
			self.x_var[1, :-1] 	>= -ALPHA_MAX - self.epsilon_var[0, :],
			self.u_var			<= +np.deg2rad(DELTA_1_MAX),
			self.u_var 			>= -np.deg2rad(DELTA_1_MAX)
		]

		# Add slack cost

		S = 10.0 / np.deg2rad(ALPHA_MAX)**2
		for i in range(self.N):
			terminalCost += S * cp.norm1(self.epsilon_var[:, i])

		# Return cost and constraints

		return terminalCost, constraints
