import numpy as np
import control
import cvxpy
import time
import matplotlib.pyplot as plt

import tinympc


""" 
Linearized cartpole example taken from 
https://github.com/TinyMPC/TinyMPC/blob/main/examples/cartpole_example.cpp 
"""
n = 4
m = 1

A_dyn = np.array([
    [1.0, 0.01, 0.0, 0.0], 
    [0.0, 1.0, 0.039, 0.0], 
    [0.0, 0.0, 1.002, 0.01] , 
    [0.0, 0.0, 0.458, 1.002]
])


B_dyn = np.array([
    [0.0], 
    [0.02], 
    [0.0], 
    [0.067]
])

Q = np.array([
    [10., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 10., 0.],
    [0., 0., 0., 1.],
])

R = np.array([
    [1.]
])

def sanity_check_DARE():


    lqr = tinympc.LQR(Q, R, A_dyn, B_dyn, N=np.inf)
    P, _, _ = control.dare(A_dyn, B_dyn, Q, R)
    print(
        np.max(np.abs(P - lqr.P(0)))
    )

def sanity_check_finite_horizon():
    N = 50
    qs = np.random.randn(N+1, n)
    rs = np.random.randn(N, m)
    xinit = np.random.randn(n)

    tic = time.time()

    lqr = tinympc.LQR(Q, R, A_dyn, B_dyn, N=N)
    lqr.compute_dp(qs, rs)
    xs, us = lqr.forward_pass(xinit)
    toc = time.time()

    toc = time.time()
    lqr_time = toc - tic
    

    assert len(lqr.Ps) == N+1
    assert len(lqr.ps) == N+1
    assert len(lqr.Ks_coef) == N
    assert len(lqr.AmBKs) == N
    assert len(lqr.Ks) == N
    assert len(lqr.ds) == N
    assert len(xs) == N+1
    assert len(us) == N

    xs_cvx = cvxpy.Variable([N+1, n])
    us_cvx = cvxpy.Variable([N, m])

    tic = time.time()
    objective = cvxpy.quad_form(xs_cvx[N], 0.5 * Q) + xs_cvx[N] @ qs[N] 
    constraints = [xs_cvx[0] == xinit]
    for i in range(N):
        objective += cvxpy.quad_form(xs_cvx[i], 0.5 * Q) + xs_cvx[i] @ qs[i] 
        objective += cvxpy.quad_form(us_cvx[i], 0.5 * R) + us_cvx[i] @ rs[i]
        constraints.append(
            xs_cvx[i+1] == A_dyn @ xs_cvx[i] + B_dyn @ us_cvx[i]
        )

    problem = cvxpy.Problem(
        cvxpy.Minimize(objective),
        constraints=constraints
    )

    problem.solve(cvxpy.OSQP, eps_abs=1e-5, eps_rel=1e-5)

    toc = time.time()
    cvx_time = toc - tic

    print("x diff ", np.max(np.abs(xs - xs_cvx.value)))
    print("u diff", np.max(np.abs(us - us_cvx.value)))


    obj_lqr = 0.5 * np.sum( (xs @ Q) * xs ) + np.sum(xs * qs) \
            + 0.5 * np.sum( (us @ R) * us ) + np.sum(us * rs)
    
    print(obj_lqr - problem.value)

    print("LQR time %.5f, CVX time %.5f" % (lqr_time, cvx_time))
    


def sanity_check_tinympc():
    
    N = 500
    uamp = 1.
    xamp = 1.

    xinit = np.array([0.5, 0, 0, 0])
    xmin = -xamp * np.ones([n,])
    xmax = xamp * np.ones([n,])
    umin = -uamp * np.ones([m,])
    umax = uamp * np.ones([m,])


    tic = time.time()
    xs_cvx = cvxpy.Variable([N+1, n])
    us_cvx = cvxpy.Variable([N, m])

    tic = time.time()
    objective = cvxpy.quad_form(xs_cvx[N], 0.5 * Q)
    constraints = [xs_cvx[0] == xinit, xmin <= xs_cvx[-1],  xs_cvx[-1]<= xmax]
    for i in range(N):
        objective += cvxpy.quad_form(xs_cvx[i], 0.5 * Q)
        objective += cvxpy.quad_form(us_cvx[i], 0.5 * R)
        constraints.append(
            xs_cvx[i+1] == A_dyn @ xs_cvx[i] + B_dyn @ us_cvx[i]
        )
        constraints.append( xmin <= xs_cvx[i])
        constraints.append( xs_cvx[i] <= xmax )
        constraints.append( umin <= us_cvx[i] )
        constraints.append( us_cvx[i] <= umax )

    problem = cvxpy.Problem(
        cvxpy.Minimize(objective),
        constraints=constraints
    )
    # problem.solve(cvxpy.MOSEK)
    problem.solve(cvxpy.OSQP, eps_abs=1e-5, eps_rel=1e-5)
    toc = time.time()
    cvx_time = toc - tic


    tic = time.time()
    xs, us = tinympc.tiny_mpc(Q, R, A_dyn, B_dyn, N, xinit, xmin, xmax, umin, umax, rho=1.,  eps=1e-5)
    toc = time.time()
    tinympc_time = toc - tic

    print("x diff ", np.max(np.abs(xs - xs_cvx.value)))
    print("u diff", np.max(np.abs(us - us_cvx.value)))

    obj_tinympc = 0.5 * np.sum( (xs @ Q) * xs ) \
            + 0.5 * np.sum( (us @ R) * us )
    
    print(obj_tinympc - problem.value)
    print("TinyMPC time %.5f, CVX time %.5f" % (tinympc_time, cvx_time))

    plt.plot(xs[:,0], 'r-')
    plt.plot(xs[:,2], 'b-')
    plt.show()




if __name__ == '__main__':
    np.random.seed(618)

    # sanity_check_DARE()
    print("----------------sanity_check_finite_horizon()----------------")
    sanity_check_finite_horizon()

    print("----------------sanity_check_tinympc()----------------")
    sanity_check_tinympc()

    