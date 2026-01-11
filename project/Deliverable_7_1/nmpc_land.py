import numpy as np
import casadi as ca
from typing import Tuple
from src.rocket import Rocket
from control import dlqr
from scipy.signal import cont2discrete

# Input constraints

DELTA_1_MAX 	= 15.0	# °
DELTA_2_MAX 	= 15.0	# °
P_AVG_MIN 		= 40.0	# %
P_AVG_MAX 		= 80.0	# %
P_DIFF_MAX 		= 20.0	# %

# Typical states used for stage cost

OMEGA_ALPHA_TYP = 30.0	# °/s
OMEGA_BETA_TYP 	= 30.0	# °/s
OMEGA_GAMMA_TYP = 60.0	# °/s

ALPHA_TYP 		= 10.0	# °
BETA_TYP 		= 10.0	# °
GAMMA_TYP 		= 1.0	# °

V_X_TYP 		= 1.0	# m/s
V_Y_TYP 		= 1.0	# m/s
V_Z_TYP 		= 2.0	# m/s

X_TYP			= 0.1	# m
Y_TYP			= 0.1	# m
Z_TYP			= 0.1	# m

# Typical inputs used for stage cost

DELTA_1_TYP 	= 10.0	# °
DELTA_2_TYP 	= 10.0	# °
P_AVG_TYP 		= 20.0	# %
P_DIFF_TYP 		= 30.0	# %

# Stage costs

Q = np.diag([
	1.0 / np.deg2rad(OMEGA_ALPHA_TYP)**2,   # omega cost
	1.0 / np.deg2rad(OMEGA_BETA_TYP)**2,    # omega cost
	1.0 / np.deg2rad(OMEGA_GAMMA_TYP)**2,   # omega cost

	1.0 / np.deg2rad(ALPHA_TYP)**2,		# varphi cost
	1.0 / np.deg2rad(BETA_TYP)**2,		# varphi cost
	1.0 / np.deg2rad(GAMMA_TYP)**2,		# varphi cost

	1.0 / V_X_TYP**2,					# v cost
	1.0 / V_Y_TYP**2,					# v cost
	1.0 / V_Z_TYP**2,					# v cost

	1.0 / X_TYP**2,						# p cost
	1.0 / Y_TYP**2,						# p cost
	1.0 / Z_TYP**2						# p cost
])
R = np.diag([
	1.0 / np.deg2rad(DELTA_2_TYP)**2,	# delta_1 cost
	1.0 / np.deg2rad(DELTA_2_TYP)**2,	# delta_2 cost
	1.0 / np.deg2rad(DELTA_2_TYP)**2,	# P_avg cost
	1.0 / np.deg2rad(DELTA_2_TYP)**2	# P_diff cost
])

class NmpcCtrl:

	NX = 12
	NU = 4

	def __init__(self, rocket: Rocket, H: float): 
		
		# Nonlinear dynamics, horizon length and sampling period

		self.f = lambda x, u: rocket.f_symbolic(x, u)[0]
		self.Ts = rocket.Ts
		self.N = int(H / self.Ts)
		
		# Linearize and discretize system around origin

		x_trim = np.zeros(self.NX)
		xs, us = rocket.trim(x_trim)
		A, B = rocket.linearize(xs, us)
		A, B = self._discretize(A, B, self.Ts)

		# Compute terminal cost
		_, self.P, _ = dlqr(A, B, Q, R)
		self.steady_input = us

	@staticmethod
	def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
		NX, NU = B.shape
		C = np.zeros((1, NX))
		D = np.zeros((1, NU))
		A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
		return A_discrete, B_discrete

	def _setup_controller(self, xt: np.ndarray) -> None:

		# Create optimization problem

		self.opti = ca.Opti()

		# Declare optimization variables and parameters

		self.x = self.opti.variable(self.NX, self.N + 1)
		self.dx = self.opti.variable(self.NX, self.N + 1)
		self.xt = self.opti.parameter(self.NX, 1)
		self.x0 = self.opti.parameter(self.NX, 1)

		self.u = self.opti.variable(self.NU, self.N)
		self.du = self.opti.variable(self.NU, self.N)
		self.us = self.opti.parameter(self.NU, 1)

		self.opti.set_value(self.xt, xt.reshape(self.NX, 1))
		self.opti.set_value(self.us, self.steady_input)

		# Define optimization cost

		cost = 0
		for k in range(self.N):
			cost += self.dx[:, k].T @ Q @ self.dx[:, k]
			cost += self.du[:, k].T @ R @ self.du[:, k]
		cost += self.dx[:, -1].T @ self.P @ self.dx[:, -1]
		self.opti.minimize(cost)

		# Define delta formulation

		self.opti.subject_to(self.dx == self.x - self.xt)
		self.opti.subject_to(self.du == self.u - self.us)
		self.opti.subject_to(self.x[:, 0] == self.x0[:, 0])

		# Define discretized dynamics

		def f_rk4(k: int):
			k1 = self.Ts * self.f(self.x[:, k], self.u[:, k])
			k2 = self.Ts * self.f(self.x[:, k] + k1 / 2, self.u[:, k])
			k3 = self.Ts * self.f(self.x[:, k] + k2 / 2, self.u[:, k])
			k4 = self.Ts * self.f(self.x[:, k] + k3, self.u[:, k])
			return self.x[:, k] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

		# Add system dynamics

		for k in range(self.N):
			self.opti.subject_to(self.x[:, k + 1] == f_rk4(k))

		# Extract state and inputs to constrain

		beta = self.x[4, :]
		z = self.x[11, :]
		delta1 = self.u[0, :]
		delta2 = self.u[1, :]
		Pavg = self.u[2, :]
		Pdiff = self.u[3, :]

		# Euler angle singularity constraint

		self.opti.subject_to((-np.deg2rad(80) <= beta) <= np.deg2rad(80))

		# Floor constraint

		self.opti.subject_to(z > 0)

		# Input constraints

		self.opti.subject_to((-np.deg2rad(DELTA_1_MAX) <= delta1) <= np.deg2rad(DELTA_1_MAX))
		self.opti.subject_to((-np.deg2rad(DELTA_2_MAX) <= delta2) <= np.deg2rad(DELTA_2_MAX))
		self.opti.subject_to((P_AVG_MIN <= Pavg) <= P_AVG_MAX)
		self.opti.subject_to((-P_DIFF_MAX <= Pdiff) <= P_DIFF_MAX)

		# Create solver

		self.opti.solver('ipopt', {
			'expand': True,
			'print_time': False,
			'ipopt': {'sb': 'yes', 'print_level': 0, 'tol': 1e-3},
		})

	def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

		# Set parameters

		self.opti.set_value(self.x0, x0.reshape(self.NX, 1))

		# Solve optimization problem

		sol = self.opti.solve()

		# Return open loop trajectory

		x_ol = sol.value(self.x)
		u_ol = sol.value(self.u)
		t_ol = np.arange(self.N + 1) * self.Ts + t0
		u0 = u_ol[:, 0]
		return u0, x_ol, u_ol, t_ol
