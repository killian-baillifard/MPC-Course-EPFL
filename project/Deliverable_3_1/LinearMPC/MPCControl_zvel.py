from .MPCControl_base import MPCControl_base
import numpy as np
import cvxpy as cp
from cvxpy import Expression, Constraint

P_AVG_MIN 		= 40.0	# %
P_AVG_MAX 		= 80.0	# %

V_Z_TYP 		= 1.0	# m/s
DP_AVG_TYP 		= 10.0	# %

class MPCControl_zvel(MPCControl_base):

	x_ids = np.array([8])
	u_ids = np.array([2])

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.diag([
			1.0 / V_Z_TYP**2		# v_z cost
		])
		R = np.diag([
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
