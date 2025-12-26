from .MPCControl_base import MPCControl_base
import numpy as np
import cvxpy as cp
from cvxpy import Expression, Constraint

class MPCControl_zvel(MPCControl_base):

	x_ids = np.array([8])
	u_ids = np.array([2])

	def _get_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:
		
		# Define stage cost
		Q = np.diag([5.0])
		R = np.diag([0.5])

		# Define trajectory cost
		cost = 0
		for i in range(self.N):
			cost += cp.quad_form(self.dx_var[:, i], Q)
			cost += cp.quad_form(self.du_var[:, i], R)
		cost += cp.quad_form(self.dx_var[:, -1], Q)

		# Define constraints
		constraints = [

			# Dynamics with delta formulation
			self.dx_var			== self.x_var - self.xs_par,
			self.du_var			== self.u_var - self.us_par,
			self.dx_var[:, 0] 	== self.dx0_par,
			self.dx_var[:, 1:] 	== self.A @ self.dx_var[:, :-1] + self.B @ self.du_var,

			# Input constraints
			self.u_var			>= 40.0,	# P_avg >= 40% 
			self.u_var			<= 80.0		# P_avg <= 80%
		]
	
		# Return cost and constraints
		return cost, constraints
