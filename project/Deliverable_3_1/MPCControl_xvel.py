from .MPCControl_base import MPCControl_base
import numpy as np
from control import dlqr
import cvxpy as cp
from cvxpy import Expression, Constraint
from mpt4py import Polyhedron

class MPCControl_xvel(MPCControl_base):

	x_ids = np.array([1, 4, 6])
	u_ids = np.array([1])

	def _get_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:
		
		# Define stage cost
		Q = np.diag([1.0, 1.0, 1.0])
		R = np.diag([1.0])

		# Compute terminal controller
		K, Qf, _ = dlqr(self.A, self.B, Q, R)
		K = -K

		# Define trajectory cost
		cost = 0
		for i in range(self.N):
			cost += cp.quad_form(self.dx_var[:, i], Q)
			cost += cp.quad_form(self.du_var[:, i], R)
		cost += cp.quad_form(self.dx_var[:, -1], Qf)
		
		# Define state constraints
		F = np.array([
			[0.0, +1.0, 0.0], 		# beta <= +10°
			[0.0, -1.0, 0.0] 		# beta >= -10°
		])
		f = np.array([
			np.pi / 18.0,			# beta <= +10°
			np.pi / 18.0			# beta >= -10°
		])
		X = Polyhedron.from_Hrep(F, f)
		
		# Define input constraints
		G = np.array([
			[+1.0],					# delta_2 <= +15°
			[-1.0]					# delta_2 >= -15°
		])
		g = np.array([
			np.pi / 12.0,			# delta_2 <= +15°
			np.pi / 12.0			# delta_2 >= -15°
		])
		U = Polyhedron.from_Hrep(G, g)

		# Compute max invariant set
		A_cl = self.A + self.B @ K
		O = X.intersect(Polyhedron.from_Hrep(U.A @ K, U.b))
		O = self._max_invariant_set(O, A_cl)

		# Define constraints
		constraints = [

			# Dynamics with delta formulation
			self.dx_var					== self.x_var - self.xs_par,
			self.du_var					== self.u_var - self.us_par,
			self.dx_var[:, 0] 			== self.dx0_par[:, 0],
			self.dx_var[:, 1:] 			== self.A @ self.dx_var[:, :-1] + self.B @ self.du_var,

			# State, input and terminal constraints
			X.A @ self.x_var[:, :-1] 	<= X.b.reshape(-1, 1),	# State lies in state constraints
			U.A @ self.u_var 			<= U.b.reshape(-1, 1),	# Input lies in input constraints
			O.A @ self.x_var[:, -1] 	<= O.b.reshape(-1, 1)	# Final state lies in terminal set
		]

		# Return cost and constraints
		return cost, constraints
