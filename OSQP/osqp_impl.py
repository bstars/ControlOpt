"""
Solve the QP 

min. 0.5 * x'Px + q'x
s.t. Ax <= b

with a naive implementation of OSQP (https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf)

This file is named osqp_impl.py instead of osqp.py to avoid conflict with OSQP library
"""

import numpy as np
import cvxpy
import time
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt


def solve(P, q, A, b, sig=1., rho=1., alpha=1.6, tol=1e-3):
    m, n = A.shape
    H = np.vstack([
        np.hstack([P + sig * np.eye(n), A.T]),
        np.hstack([A, -1/rho * np.eye(m)])
    ])
    Hinv = np.linalg.inv(H)

    # note that w is 0 at each iteration, so it's ignored in the paper and other implementations
    # here w is included for the sake of ADMM-ness

    x = np.zeros(n) # primal vairable
    z = np.zeros(m) # primal variable
    w = np.zeros(n) # dual variable
    y = np.zeros(m) # dual variable

    objs = []
    xs = []
    num_iter = 0

    while True:
        num_iter += 1
        objs.append(
            0.5 * x @ P @ x + q @ x
        )
        xs.append(x.copy())



        # Solve the equality constrained QP 
        rhs = np.concatenate([
            -q + sig * x - w,
            z - 1/rho * y
        ])
        xv = Hinv @ rhs

        x_tilde = xv[:n]
        v = xv[n:]
        z_tilde = z - 1/rho * y + 1/rho * v

        # Primal update
        x_next = alpha * x_tilde + (1 - alpha) * x + 1/sig * w
        z_next = alpha * z_tilde + (1 - alpha) * z + 1/rho * y
        z_next = np.minimum(z_next, b)

        # Dual update
        w_next = w + sig * (
            alpha * x_tilde + (1-alpha) * x - x_next
        )

        y_next = y + rho * (
            alpha * z_tilde + (1 - alpha) * z - z_next
        )


        x = x_next
        z = z_next
        w = w_next
        y = y_next

        # check KKT residual
        rp = A @ x - z
        rd = P @ x + q + A.T @ y

        rnorm = max(
            np.max(np.abs(rp)),
            np.max(np.abs(rd))
        )

        if rnorm < tol:
            return x, num_iter


if __name__ == '__main__':
    m = 200
    n = 400


    # construct random QP

    B = np.random.randn(n,n)
    P = B.T @ B + 0.5 * np.eye(n)
    q = np.random.randn(n)
    x0 = np.random.randn(n)
    A = np.random.randn(m,n)
    b = A @ x0 + np.random.uniform(0, 1, [m,])


    tic = time.time()
    x_osqp, iter_osqp = solve(P, q, A, b, tol=1e-5)
    toc = time.time()
    time_osqp = toc - tic

    tic = time.time()
    xcvx = cvxpy.Variable([n,])
    problem = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.quad_form(xcvx, 0.5 * P) + q @ xcvx),
        [A @ xcvx <= b]
    )
    problem.solve(cvxpy.OSQP)
    xcvx = xcvx.value
    toc = time.time()
    time_cvx = toc - tic

    print("%d OSQP iterations" % iter_osqp)
    print("x diff: %.10f" % np.max(np.abs(xcvx - x_osqp)))
    print("CVXPY time: %.4f, OSQP time: %.4f" % (time_cvx, time_osqp))
