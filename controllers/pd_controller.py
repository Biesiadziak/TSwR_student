import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot, q_d_ddot):
        ### TODO: Please implement me
        u = self.kp * (q - q_d) + self.kd * (q_dot - q_d_dot)
        return u
