from .MPCControl_base import MPCControl_base
import numpy as np
import cvxpy as cp
from cvxpy import Expression, Constraint

P_DIFF_MAX 		= 20.0 				# %

OMEGA_GAMMA_TYP = 30.0 				# °/s
GAMMA_TYP 		= 1.0 				# °
P_DIFF_TYP 		= P_DIFF_MAX / 2.0 	# %

class MPCControl_roll(MPCControl_base):

	x_ids = np.array([2, 5])
	u_ids = np.array([3])

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.diag([
			1.0 / np.deg2rad(OMEGA_GAMMA_TYP)**2,	# omega_gamma cost
			1.0 / np.deg2rad(GAMMA_TYP)**2			# gamma cost
		])
		R = np.diag([
			1.0 / P_DIFF_TYP**2						# P_diff cost
		])
		return Q, R

	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:

		# Define terminal cost

		Q, _ = self._get_stage_cost()
		terminalCost = cp.quad_form(self.dx_var[:, -1], Q)

		# Define constraints

		constraints = [
			self.u_var	>= -P_DIFF_MAX,	# P_diff >= -20%
			self.u_var	<= +P_DIFF_MAX	# P_diff <= +20%
		]
	
		# Return cost and constraints

		return terminalCost, constraints
