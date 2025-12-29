from .MPCControl_base import MPCControl_base
import numpy as np
from control import dlqr
import cvxpy as cp
from cvxpy import Expression, Constraint
from mpt4py import Polyhedron

class MPCControl_xvel(MPCControl_base):

	x_ids = np.array([1, 4, 6])
	u_ids = np.array([1])

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.diag([5e-1, 5e-1, 5e-1])
		R = np.diag([5e1])
		return Q, R

	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:

		# Compute terminal controller
		Q, R = self._get_stage_cost()
		K, Qf, _ = dlqr(self.A, self.B, Q, R)
		K = -K

		# Define terminal cost
		terminalCost = cp.quad_form(self.dx_var[:, -1], Qf)
		
		# Define state constraints
		F = np.array([
			[0.0, +1.0, 0.0], 		# beta <= +10°
			[0.0, -1.0, 0.0] 		# beta >= -10°
		])
		f = np.array([
			np.deg2rad(10.0),		# beta <= +10°
			np.deg2rad(10.0)		# beta >= -10°
		])
		X = Polyhedron.from_Hrep(F, f)
		
		# Define input constraints
		G = np.array([
			[+1.0],					# delta_2 <= +15°
			[-1.0]					# delta_2 >= -15°
		])
		g = np.array([
			np.deg2rad(15.0),		# delta_2 <= +15°
			np.deg2rad(15.0)		# delta_2 >= -15°
		])
		U = Polyhedron.from_Hrep(G, g)

		# Compute max invariant set
		A_cl = self.A + self.B @ K
		O = X.intersect(Polyhedron.from_Hrep(U.A @ K, U.b))
		O = self._max_invariant_set(O, A_cl)
		self.O_inf = O

		# Define constraints
		constraints = [
			X.A @ self.x_var[:, :-1]	<= X.b.reshape(-1, 1),	# State lies in state constraints
			U.A @ self.u_var			<= U.b.reshape(-1, 1),	# Input lies in input constraints
			O.A @ self.x_var[:, -1]		<= O.b.reshape(-1, 1)	# Final state lies in terminal set
		]

		# Return cost and constraints
		return terminalCost, constraints
