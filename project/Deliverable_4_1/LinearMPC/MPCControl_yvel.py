from .MPCControl_base import MPCControl_base
import numpy as np
from control import dlqr
import cvxpy as cp
from cvxpy import Expression, Constraint
from mpt4py import Polyhedron

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
		
		# Compute terminal controller

		Q, R = self._get_stage_cost()
		K, Qf, _ = dlqr(self.A, self.B, Q, R)
		K = -K
		terminalCost = cp.quad_form(self.dx_var[:, -1], Qf)
		
		# Define state constraints

		F = np.array([
			[0.0, +1.0, 0.0], 			# alpha <= +10°
			[0.0, -1.0, 0.0] 			# alpha >= -10°
		])
		f = np.array([
			np.deg2rad(ALPHA_MAX),		# alpha <= +10°
			np.deg2rad(ALPHA_MAX)		# alpha >= -10°
		])
		X = Polyhedron.from_Hrep(F, f)
		
		# Define input constraints

		G = np.array([	
			[+1.0],						# delta_1 <= +15°
			[-1.0]						# delta_1 >= -15°
		])
		g = np.array([
			np.deg2rad(DELTA_1_MAX),	# delta_1 <= +15°
			np.deg2rad(DELTA_1_MAX)		# delta_1 >= -15°
		])
		U = Polyhedron.from_Hrep(G, g)

		# Compute max invariant set

		A_cl = self.A + self.B @ K
		O = X.intersect(Polyhedron.from_Hrep(U.A @ K, U.b))
		O = self._max_invariant_set(O, A_cl, self.N)
		self.O_inf = O

		# Define constraints with slack variable

		self.epsilon_var = cp.Variable((f.size, self.N), 'epsilon', nonneg=True)
		constraints = [
			X.A @ self.x_var[:, :-1]	<= X.b.reshape(-1, 1) + self.epsilon_var,	# State penalized for violating constraints
			U.A @ self.u_var			<= U.b.reshape(-1, 1),						# Input lies in input constraints
			O.A @ self.x_var[:, -1]		<= O.b.reshape(-1, 1)						# Final state lies in terminal set
		]

		# Add slack cost

		S = 10.0 / np.deg2rad(ALPHA_MAX)**2
		for i in range(self.N):
			terminalCost += S * cp.norm1(self.epsilon_var[:, i])

		# Return cost and constraints

		return terminalCost, constraints
