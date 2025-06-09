import numpy as np

#from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel
# from models.ideal_model import IdealModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        p1 = p[0]
        p2 = p[1]
        self.L = np.array([
            [3 * p1, 0],
            [0, 3 * p2],
            [3 * p1 ** 2, 0],
            [0, 3 * p2 ** 2],
            [p1 ** 3, 0],
            [0, p2 ** 3]
        ])
        W = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        M_hat = self.model.M(q0)
        C_hat = self.model.C(q0)
        A = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        A[2:4, 2:4] = -np.linalg.inv(M_hat) @ C_hat
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        B[2:4, :2] = np.linalg.inv(M_hat)
        print(A)
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        
        x = np.hstack([q, q_dot])

        M_hat = self.model.M(x)        # 2x2 macierz inercji w pozycji q
        C_hat = self.model.C(x) 

        A = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        A[2:4, 2:4] = -np.linalg.inv(M_hat) @ C_hat
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        B[2:4, :2] = np.linalg.inv(M_hat)

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        
        q = x[0:2]
        q_dot = x[2:4] 

        z_hat = self.eso.get_state()

        q_hat = z_hat[0:2]
        q_dot_hat = z_hat[2:4]
        f_hat = z_hat[4:6]

        e = q_d - q
        e_dot = q_d_dot - q_dot_hat

        v = self.Kp @ e + self.Kd @ e_dot + q_d_ddot

        u = self.model.M(x) @ (v - f_hat) + self.model.C(x) @ q_dot_hat

        self.update_params(q_hat, q_dot_hat)

        self.eso.update(np.array(q).reshape(-1, 1), u.reshape(-1, 1))

        return u
