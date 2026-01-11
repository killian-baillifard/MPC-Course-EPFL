from .OffsetFreeMPCControl_base import OffsetFreeMPCControl_base
import numpy as np
import cvxpy as cp
from cvxpy import Expression, Constraint
from scipy.signal import place_poles

P_AVG_MIN 		= 40.0	# %
P_AVG_MAX 		= 80.0	# %

V_Z_TYP 		= 1.0	# m/s
DP_AVG_TYP 		= 10.0	# %

class MPCControl_zvel(OffsetFreeMPCControl_base):

	x_ids = np.array([8])
	u_ids = np.array([2])
	z_hat_initialized = False

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
	
	def _setup_estimator(self) -> np.ndarray:

		# Define measurable states and disturbance dynamics

		ND = 1
		NY = 1
		self.C = np.array([[1]])
		self.Cd = np.array([[0]])
		self.Bd = self.B

		# Compute augmented matrices

		self.A_hat = np.vstack((
			np.hstack((self.A, self.Bd)),
			np.hstack((np.zeros((NY, self.NX)), np.eye(NY)))
		))
		self.B_hat = np.vstack((self.B, np.zeros((ND, self.NU))))
		self.C_hat = np.hstack((self.C, self.Cd))

		# Compute gain by pole placement

		poles = np.array([0.5, 0.7])
		K = place_poles(self.A_hat.T, self.C_hat.T, poles)
		self.L = -K.gain_matrix.T

		# Return disturbance dynamics

		return self.Bd
	
	def _compute_steady_state(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

		xd = cp.Variable((self.NX, 1), name="xd")   # deviation state
		ud = cp.Variable((self.NU, 1), name="ud")   # deviation input
		
		constraints = [
			ud >= P_AVG_MIN - self.us,
			ud <= P_AVG_MAX - self.us,
			xd == self.A @ xd + self.B @ ud + self.Bd @ self.d_hat,
			r  == self.C @ xd + self.Cd @ self.d_hat
		]

		cost = cp.quad_form(ud, np.eye(self.NU))
		prob = cp.Problem(cp.Minimize(cost), constraints)
		prob.solve()

		if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
			raise RuntimeError(f"Steady-state QP infeasible: {prob.status}")

		# convert to absolute steady state for the MPC shift parameters
		xs = self.xs + xd.value
		us = self.us + ud.value
		return xs.reshape(self.NX, 1), us.reshape(self.NU, 1)

	def _update_estimator(self, x0: np.ndarray, u_prev: np.ndarray) -> np.ndarray:

		# Initialize estimated state with first measurement

		if not self.z_hat_initialized:
			self.z_hat = np.vstack([
				x0.reshape(self.NX, 1),
				np.zeros((self.NU, 1))
			])
			self.z_hat_initialized = True
			return np.zeros((self.NU, 1))

		# Estimate state and disturbance

		y_meas = x0.reshape(self.NX, 1)
		du_prev = u_prev - self.us
		self.z_hat = (self.A_hat @ self.z_hat + self.B_hat @ du_prev + self.L @ (self.C_hat @ self.z_hat - y_meas))

		# ud_min = P_AVG_MIN - float(self.us)
		# ud_max = P_AVG_MAX - float(self.us)
		# d_min = -ud_max
		# d_max = -ud_min
		# d_hat = float(self.z_hat[self.NX, 0])
		# d_hat = np.clip(d_hat, d_min, d_max)
		# self.z_hat[self.NX, 0] = d_hat

		return self.z_hat[self.NX:]
