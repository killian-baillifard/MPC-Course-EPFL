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
    N = 20

    def __init__(self, rocket: Rocket):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x,u: rocket.f_symbolic(x,u)[0]

    def _setup_controller(self) -> None:

        self.ocp = ca.Opti()

        # Variables
        self.x = self.ocp.variable(self.NX, self.N + 1)
        self.u = self.ocp.variable(self.NU, self.N)
        self.dx = self.ocp.variable(self.NX, self.N + 1)
        self.xt = self.ocp.variable(self.NX, self.N + 1)
        self.x0 = self.ocp.parameter(self.NX, 1)

        # Extract states to optimize or constrain
        self.e = self.dx[9:12, :]
        self.beta = self.x[4, :]
        self.z = self.x[11, :]

        # Optimization problem
        self.ocp.minimize(

            # Reduce position error
            self.e @ self.e.T
        )

        # Initial state
        self.ocp.subject_to(self.x[:, 0] == self.x0[:, 0])

        # System dynamics
        for k in range(self.N):
            self.ocp.subject_to(self.x[:, k + 1] == self.f(self.x[:, k], self.u[:, k]))

        # Euler angle singularity constraint
        self.ocp.subject_to(self.beta > np.deg2rad(80))

        # Floor constraint
        self.ocp.subject_to(self.z > 0)

        # set solver
        options = {
            "print_time": False,
            "ipopt": {"sb": "yes", "print_level": 0, "tol": 1e-3},
        }
        self.ocp.solver("ipopt", options)

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        u0 = ...
        x_ol = ...
        u_ol = ...
        t_ol = ... 

        return u0, x_ol, u_ol, t_ol