from .TubeMPCControl_base import TubeMPCControl_base
import numpy as np
from control import dlqr
import cvxpy as cp
from mpt4py import Polyhedron
from cvxpy import Expression, Constraint

P_AVG_MIN 		= 40.0	# %
P_AVG_MAX 		= 80.0	# %

V_Z_TYP 		= 1.3	# m/s
P_Z_TYP			= 0.5	# m
DP_AVG_TYP 		= 16.0	# %

LAMBDA = 0.0

""" Smoothest no noise
V_Z_TYP 		= 1.3	# m/s
P_Z_TYP			= 0.5	# m
DP_AVG_TYP 		= 16	# %

LAMBDA = 0.0
"""

""" Works on all noise settings
V_Z_TYP 		= 1.3	# m/s
P_Z_TYP			= 0.5	# m
DP_AVG_TYP 		= 18.0	# %

LAMBDA = 0.0
"""

""" Good results on random
V_Z_TYP 		= 1.0	# m/s
P_Z_TYP			= 0.25	# m
DP_AVG_TYP 		= 20.0	# %

LAMBDA = -1.0
"""

# Disturbance bounds (file-level constants)
W_MIN = -15.0
W_MAX = 5.0

class MPCControl_z(TubeMPCControl_base):
	"""
	Tube MPC for z-subsystem, with states [v_z, p_z] and input [P_avg].

	Assumed dynamics (continuous and hence discrete):
		xdot = A x + B u + B w

	Constraints (in delta coords):
	- p_z >= 0 (collision avoidance, robustified through tightening)
	- P_avg in [40%, 80%] as absolute input (converted to delta du bounds)
	- disturbance w in [W_MIN, W_MAX]
	"""

	x_ids: np.ndarray = np.array([8, 11])  # [v_z, p_z]
	u_ids: np.ndarray = np.array([2])      # [P_avg]

	def _get_stage_cost(self) -> tuple[np.ndarray, np.ndarray]:
		Q = np.exp(LAMBDA) * np.diag([
			1.0 / V_Z_TYP**2,		# v_z cost
			1.0 / P_Z_TYP**2		# p_z cost
		])
		R = np.exp(-LAMBDA) * np.diag([
			1.0 / DP_AVG_TYP**2		# delta P_avg cost
		])
		return Q, R

	def _get_terminal_cost_and_constraints(self) -> tuple[Expression, list[Constraint]]:

		# ---- Compute terminal controller (LQR) and terminal cost weight ----
		Q, R = self._get_stage_cost()
		K, Qf, _ = dlqr(self.A, self.B, Q, R)
		K = -K  # because dlqr uses u=-Kx convention
		self.K = np.array(K)

		terminalCost = cp.quad_form(self.dx_var[:, -1], Qf)

		# ---- Define state constraint set X (DELTA) ----
		# state is [v_z, p_z]; collision avoidance: p_z >= 0 -> -p_z <= 0
		z_s = float(self.xs.reshape(-1)[1])  # second state in your x_ids order: p_z trim (≈3)
		F = np.array([
			[0.0, -1.0],   # -Δp_z <= z_s
		])
		f = np.array([
			z_s
		])
		X = Polyhedron.from_Hrep(F, f)

		# ---- Define input constraint set U (DELTA) ----
		# Absolute input: P_AVG_MIN <= u <= P_AVG_MAX, with u = us + du
		us = float(self.us.reshape(-1)[0])
		du_min = P_AVG_MIN - us
		du_max = P_AVG_MAX - us

		# H-rep: A u <= b
		# du >= du_min -> -du <= -du_min
		# du <= du_max ->  du <=  du_max
		G = np.array([
			[-1.0],
			[+1.0],
		])
		g = np.array([
			-du_min,
			+du_max,
		])
		U = Polyhedron.from_Hrep(G, g)

		# ---- Define disturbance set W ----
		# w ∈ [W_MIN, W_MAX]
		H = np.array([
			[+1.0],   # w <= W_MAX
			[-1.0],   # -w <= -W_MIN  <=> w >= W_MIN
		])
		h = np.array([
			W_MAX,
			-W_MIN,
		])
		W = Polyhedron.from_Hrep(H, h)

		# ---- Minimal robust invariant set E for error dynamics ----
		# Error: e+ = (A + B K) e + B w
		A_cl = self.A + self.B @ self.K
		BW = self.B @ W                       # <-- KEY FIX: use B (same matrix as input)
		E = self._min_robust_invariant_set(A_cl, BW, self.N)
		self.E = E

		# ---- Tighten constraints ----
		# X_tilde = X ⊖ E   and   U_tilde = U ⊖ K E
		X_tilde = X - E
		U_tilde = U - (self.K @ E)

		self.X_tilde = X_tilde
		self.U_tilde = U_tilde

		# ---- Terminal tightened set Xf_tilde ----
		# Need dz ∈ X_tilde and K dz ∈ U_tilde
		# Encode K dz ∈ U_tilde as U_tilde.A (K dz) <= U_tilde.b
		X_and_KU = X_tilde.intersect(Polyhedron.from_Hrep(U_tilde.A @ self.K, U_tilde.b))
		Xf_tilde = self._max_invariant_set(X_and_KU, A_cl, self.N)
		self.Xf_tilde = Xf_tilde

		# ---- Build CVXPY constraints ----
		# self.epsilon_var = cp.Variable((1, self.N + 1), 'epsilon', nonneg=True)
		constraints = [
			# self.dx_var[1, :]				>= self.xs_cst[1, 0] + self.epsilon_var,
			X_tilde.A @ self.dx_var[:, :-1] <= X_tilde.b.reshape(-1, 1),
			U_tilde.A @ self.du_var 		<= U_tilde.b.reshape(-1, 1),
			Xf_tilde.A @ self.dx_var[:, -1] <= Xf_tilde.b.reshape(-1, 1)
		]

		# Add slack cost
		# S = 10.0 / np.deg2rad(P_Z_TYP)**2
		# for i in range(self.N):
		# 	terminalCost += S * cp.norm1(self.epsilon_var[:, i])

		return terminalCost, constraints