
"""
Solve constrained MPC 

    min{x,u}.   \sum_{t=0}^{N-1}   0.5 * x[t]*Q*x[t] + q[t]*x[t] 
                            + 0.5 * u[t]*R*u[t] + r[t]*u[t]
                + 0.5 * x[N]*Q*x[N] + q[N]*x[N] 

    s.t.        x[t+1] = A * x[t] + B * u[t]  for t=0,...N-1
                x[0] = x_init
                x[t] \in X for t=1,...,N
                u[t] \in U for t=0,...,N-1

with a naive implementation of TinyMPC (https://arxiv.org/pdf/2310.16985)

Some notations are different than in the paper, please check the detailed derivation at lovinglavigne.com
"""


import numpy as np
import control


class LQR():
    def __init__(self, Q, R, A, B, N):

        """
        This class solves the LQR problem
            min{x,u}.   \sum_{t=0}^{N-1}    0.5 * x[t]*Q*x[t] + q[t]*x[t] 
                                            + 0.5 * u[t]*R*u[t] + r[t]*u[t]
                        + 0.5 * x[N]*Q*x[N] + q[N]*x[N] 

            s.t.        x[t+1] = A @ x[t] + B @ u[t] for t=0,...N-1
                        x[0] = x_init

            ( Since x[0] is give, sometime we ignore the objective related to x[0] in computation )

        The cost function for each subproblem is 
            Jn(x) = 0.5 * x*P_n*x + q_n*x + gamma_n

        params:
            Q: Quadratic terms for x[t]
            R: Quadratic terms for u[t]
            A(B):
                The dynamic matrices x[t+1] = A @ x[t] + B @ u[t]
            N: Time horizon, can be int or np.inf

            
            for time index, we use the convention shown as follows
            
                            u[0] = -K[0]x[0] - d[0]
            x[0] P[0] p[0]  ------------------------------> x[1] P[1] p[1]

                            u[1] = -K[1]x[1] - d[1]
                            ------------------------------> x[2] P[2] p[2]

                            ...

                            u[N-1] = -K[N-1]x[N-1] - d[N-1]
                            ------------------------------> x[N] P[N] p[N]
        """
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.N = N
        self.n, self.m = B.shape

        # self.Ps has shape [N+1, n, n]
        # self.Ks has shape [N, n, m]
        self.Ps, self.Ks, self.Ks_coef, self.AmBKs = self.compute_riccati()

        # self.ps has shape [N+1, n]
        # self.ds has shape [N, m]
        self.ps, self.ds = None, None



    def compute_riccati(self):
        """
        Compute the solution to the Discrete Algebraic Riccati Equation (DARE)
        Since P's and K's only depends on [Q,R,A,B], this function should be called only onces from TinyMPC
        """

        Ps = [self.Q.copy()]
        Ks = []
        Ks_coef = [] # we save this for computation of ds in compute_dp()
        AmBKs = []

        num_iter = 0
        while True:

            num_iter += 1

            BtP = self.B.T @ Ps[-1]
            K_coef = np.linalg.inv(self.R + BtP @ self.B)
            K = K_coef @ (BtP @ self.A)
            AmBK = self.A - self.B @ K

            Ks.append(K.copy())
            Ks_coef.append(K_coef.copy())
            AmBKs.append(AmBK.copy())


            P_next = self.Q + K.T @ self.R @ K + AmBK.T @ Ps[-1] @ AmBK
            Ps.append(P_next.copy())

            if self.N == np.inf and np.max(np.abs(Ps[-1] - Ps[-2])) <= 1e-6:
                print("DARE converges in %d iterations" % num_iter)
                return [P_next], [K], [K_coef], [AmBK]

            if self.N != np.inf and num_iter == self.N:
                return Ps[::-1], Ks[::-1], Ks_coef[::-1], AmBKs[::-1]



    def compute_dp(self, qs, rs):
        """
        Compute ds and ps for LQR, this function is called multiple times from TinyMPC
        qs: np.array with shape [N+1, n]
        rs: np.array with shape [N, m]
        """

        ds = []
        ps = [qs[-1]]

        for n in range(self.N-1, -1, -1):
            ds.append( 
                self.K_coef(n) @ (rs[n] + self.B.T @ ps[-1]) 
            )
            ps.append(
                self.K(n).T @ (self.R @ ds[-1] - rs[n]) + qs[n] + self.AmBK(n).T @ (ps[-1] - self.P(n+1) @ self.B @ ds[-1])
            )

        self.ds = ds[::-1]
        self.ps = ps[::-1]

    def forward_pass(self, x0):
        xs = [x0]
        us = []
        
        for i in range(self.N):
            
            u = -self.K(i) @ xs[-1] - self.d(i)
            x = self.A @ xs[-1] + self.B @ u

            xs.append(x.copy())
            us.append(u.copy())

        return np.array(xs), np.array(us)


    # dummy aux functions to unify API for N=inf and N<inf
    def K(self, n): return self.Ks[0] if self.N == np.inf else self.Ks[n]
         
    def P(self, n): return self.Ps[0] if self.N == np.inf else self.Ps[n]

    def K_coef(self, n): return self.Ks_coef[0] if self.N == np.inf else self.Ks_coef[n]

    def AmBK(self, n): return self.AmBKs[0] if self.N == np.inf else self.AmBKs[n]

    def d(self, n): return self.ds[0] if self.N == np.inf else self.ds[n]

    def p(self, n): return self.ps[0] if self.N == np.inf else self.ps[n]
        
        




def tiny_mpc(Q, R, A, B, N, xinit, xmin, xmax, umin, umax, rho=1., eps=1e-3):
    """
    This class solves the LQR problem
        min{x,u}.   \sum_{t=0}^{N-1}    0.5 * x[t]*Q*x[t] + q[t]*x[t] 
                                        + 0.5 * u[t]*R*u[t] + r[t]*u[t]
                    + 0.5 * x[N]*Q*x[N] + q[N]*x[N] 

        s.t.        x[t+1] = A * x[t] + B * u[t]  for t=0,...N-1
                    x[0] = x_init
                    xmin <= x[t] <= xmax for t=1,...,N
                    umin <= u[t] <= umax for t=0,...,N-1
    with OSQP (ADMM)

    Here we assume 
        the starting point satisfied xmin <= x[0] <= xmax
        q and r are zero    
    """
    n, m = B.shape
    Q1 = Q + rho * np.eye(n)
    R1 = R + rho * np.eye(m)

    lqr = LQR(Q1, R1, A, B, N)
    lqr.compute_riccati()


    # initialize primal variable
    xs = np.zeros([N+1, n])
    xs[0] = xinit
    xs_tilde = xs.copy()

    us = np.zeros([N, m])
    us_tilde = us.copy()

    # initialize dual variable
    ys = np.zeros([N+1, n]) # remark : ys[0] is always 0
    gs = np.zeros([N, m])

    num_iter = 0
    while True:
        num_iter += 1

        # Solve ADMM subproblem with LQR
        qs = ys - rho * xs_tilde
        rs = gs - rho * us_tilde

        lqr.compute_dp(qs, rs)

        xs_new, us_new = lqr.forward_pass(xinit)

        # Solve ADMM subproblem by projection
        xs_tilde_new = np.clip(
            xs_new + 1 / rho * ys,
            xmin, 
            xmax
        )

        us_tilde_new = np.clip(
            us_new + 1/ rho * gs,
            umin, 
            umax
        )


        residual = max(
            np.max(np.abs(xs_new - xs_tilde_new)), # primal residual
            np.max(np.abs(us_new - us_tilde_new)), # primal residual
            np.max(np.abs(xs_tilde - xs_tilde_new)) * rho, # dual residual
            np.max(np.abs(us_tilde - us_tilde_new)) * rho, # dual residual
        )

        if residual <= eps:
            print(num_iter, " ADMM iterations")
            return xs_new, us_new


        # Dual update
        ys = ys + rho * (xs_new - xs_tilde_new)
        gs = gs + rho * (us_new - us_tilde_new)

        xs = xs_new
        us = us_new
        xs_tilde = xs_tilde_new
        us_tilde = us_tilde_new




