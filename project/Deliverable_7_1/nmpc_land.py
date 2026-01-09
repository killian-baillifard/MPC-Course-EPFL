import numpy as np
import casadi as ca
from typing import Tuple
from src.rocket import Rocket

class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """

    NX = 12
    NU = 4

    def __init__(self, rocket: Rocket, H: float):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]

        # Horizon length and sampling period
        self.Ts = rocket.Ts
        self.N = int(H / self.Ts)

    def _setup_controller(self, xt: np.ndarray) -> None:

        # Create optimization problem
        self.opti = ca.Opti()

        # Variables
        self.x = self.opti.variable(self.NX, self.N + 1)
        self.u = self.opti.variable(self.NU, self.N)
        self.dx = self.opti.variable(self.NX, self.N + 1)
        self.x0 = self.opti.parameter(self.NX, 1)
        self.xt = self.opti.parameter(self.NX, 1)
        self.opti.set_value(self.xt, xt.reshape(self.NX, 1))

        # Extract states to optimize
        eroll = self.dx[5, :]
        ex = self.dx[9, :]
        ey = self.dx[10, :]
        ez = self.dx[11, :]

        # Define optimization cost
        self.opti.minimize(

            # Reduce pose error
            eroll @ eroll.T +
            ex @ ex.T +
            ey @ ey.T +
            ez @ ez.T
        )

        # Delta formulation
        self.opti.subject_to(self.dx == self.x - self.xt)
        self.opti.subject_to(self.x[:, 0] == self.x0[:, 0])

        # Discretization
        def f_rk4(k: int):
            k1 = self.Ts * self.f(self.x[:, k], self.u[:, k])
            k2 = self.Ts * self.f(self.x[:, k] + k1 / 2, self.u[:, k])
            k3 = self.Ts * self.f(self.x[:, k] + k2 / 2, self.u[:, k])
            k4 = self.Ts * self.f(self.x[:, k] + k3, self.u[:, k])
            return self.x[:, k] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # System dynamics
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
        self.opti.subject_to((-np.deg2rad(15) <= delta1) <= np.deg2rad(15))
        self.opti.subject_to((-np.deg2rad(15) <= delta2) <= np.deg2rad(15))
        self.opti.subject_to((40 <= Pavg) <= 80)
        self.opti.subject_to((-20 <= Pdiff) <= 20)

        # set solver
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

        # Extract output
        x_ol = sol.value(self.x)
        u_ol = sol.value(self.u)
        t_ol = np.arange(self.N + 1) * self.Ts + t0

        # Return output
        u0 = u_ol[:, 0]
        return u0, x_ol, u_ol, t_ol