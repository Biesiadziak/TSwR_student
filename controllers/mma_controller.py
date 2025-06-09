import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3

        model1 = ManiuplatorModel(Tp, m3=0.1, r3=0.05)
        model2 = ManiuplatorModel(Tp, m3=0.01, r3=0.01)
        model3 = ManiuplatorModel(Tp, m3=1.0, r3=0.3)

        self.models = [model1, model2, model3]
        self.i = 0
        self.prev_u = np.zeros((2, 1))

    def choose_model(self, x, tau):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q = x[:2]        # shape (2,)
        q_dot = x[2:]    # shape (2,)
        errors = []

        for model in self.models:
            M = model.M(x)
            C = model.C(x)

            # Flatten tau for compatibility
            tau_vec = tau.flatten()

            # Compute acceleration
            q_ddot = np.linalg.inv(M) @ (tau_vec - C @ q_dot)

            # Predict next state using Euler integration
            q_next = q + model.Tp * q_dot
            q_dot_next = q_dot + model.Tp * q_ddot

            # Flatten into a single predicted state
            x_pred = np.concatenate([q_next, q_dot_next])

            error = np.linalg.norm(x - x_pred)

            errors.append(error)

        self.i = int(np.argmin(errors))


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        Kp = 40
        Kd = 30

        self.choose_model(x, self.prev_u)
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot + Kd * (q_r_dot - q_dot) + Kp * (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.prev_u = u
        return u
