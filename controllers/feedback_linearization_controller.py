import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x

        Kp = 34
        Kd = 22

        v = q_r_ddot + Kd * (q_r_dot - np.array([q1_dot, q2_dot])) + Kp * (q_r - np.array([q1, q2]))

        u = self.model.M(x) @ v + self.model.C(x) @ np.array([q1_dot, q2_dot])
        
        return u