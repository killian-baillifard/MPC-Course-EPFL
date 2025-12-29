from .MPCControl_base import MPCControl_base
import numpy as np
import cvxpy as cp
from cvxpy import Expression, Constraint

class MPCControl_zvel(MPCControl_base):

	x_ids = np.array([8])
	u_ids = np.array([2])

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.diag([1e0])
		R = np.diag([1e-2])
		return Q, R

	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:

		# Define terminal cost
		Q, _ = self._get_stage_cost()
		terminalCost = cp.quad_form(self.dx_var[:, -1], Q)

		# Define constraints
		constraints = [
			self.u_var	>= 40.0,	# P_avg >= 40% 
			self.u_var	<= 80.0		# P_avg <= 80%
		]
	
		# Return cost and constraints
		return terminalCost, constraints
