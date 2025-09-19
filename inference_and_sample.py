import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, lax
from functools import partial
from jax.tree_util import tree_map
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
from pymanopt.function import jax as funjax

def generate_latents_and_observations(key, u, A, B, Q, mu0, Q0, C, d, R):
    ''' 
    Parameters
    ----------
    S: number of trials/sessions
    T: number of time points in trial/session
    '''
    T = u.shape[0]
        
    key_x0, key_y0, key_step = jr.split(key, num=3)

    # first emission
    x0 = jr.multivariate_normal(key_x0, mu0, Q0)
    y0 = jr.multivariate_normal(key_y0, C @ x0 + d, R)
        
    def step(x_prev, args):
        key_step, u_prev = args
        key_x, key_y = jr.split(key_step, num=2)
        x_current = jr.multivariate_normal(key_x, A @ x_prev + B @ u_prev, Q)
        y_current = jr.multivariate_normal(key_y, C @ x_current + d, R)
        return x_current, (x_current, y_current)
        
    keys_step = jr.split(key_step, num = T - 1)
    _, (xs, ys) = lax.scan(step, init=x0, xs=(keys_step, u[:-1]))
        
    x = jnp.concatenate([x0[jnp.newaxis, :],xs])
    y = jnp.concatenate([y0[jnp.newaxis, :],ys])
        
    return x, y

# jax efficient parallelization across batches
generate_latents_and_observations_batches = jit(vmap(generate_latents_and_observations, in_axes=(0,0,None,None,None,None,None,None,None,None)))

def Kalman_filter_E_step(y, u, A, B, Q , mu0, Q0, C, d, R):
    ''' 
    Kalman filter in E-step that computes forwards: p(x_t | y_1:t) and prior p(x_t | y_1:t-1)
    
    for each trial/session individually

    note that inputs come in only at the second time step
    '''

    T = y.shape[0]
    K = A.shape[0]
    D = C.shape[0]
        
    def update_based_on_priors(mu_prior, V_prior, y_current):
            
        # normalizing factor
        norm_fact = - 0.5 * jnp.linalg.slogdet(C @ V_prior @ C.T + R)[1]
        norm_fact = norm_fact - 0.5 * (y_current - C @ mu_prior - d).T @ jnp.linalg.solve(C @ V_prior @ C.T + R, y_current - C @ mu_prior - d)
            
        # filter updates
        CRC = C.T @ jnp.linalg.solve(R, C) 
        V_current = jnp.linalg.solve(CRC + jnp.linalg.solve(V_prior, jnp.eye(K, dtype=V_prior.dtype)), jnp.eye(K, dtype=V_prior.dtype)) 
        CRy = C.T @ jnp.linalg.solve(R, y_current - d)
        mu_current = V_current @ (CRy + jnp.linalg.solve(V_prior, mu_prior)) 
            
        # to ensure precise symmetry
        V_current = 0.5 * (V_current + V_current.T)
            
        return mu_current, V_current, norm_fact
        
    def step(state_prev, args): # forward step in filtering
            
        u_prev, y_current = args
        mu_prev, V_prev = state_prev
            
        # prior updates
        mu_prior_current = A @ mu_prev + B @ u_prev
        V_prior_current = A @ V_prev @ A.T + Q
            
        # filter updates
        mu_current, V_current, norm_fact = update_based_on_priors(mu_prior_current, V_prior_current, y_current)
        
        return (mu_current, V_current), (mu_current, mu_prior_current, V_current, V_prior_current, norm_fact)
        
    # first step
    mu_0, V_0, norm_fact_0 = update_based_on_priors(mu0, Q0, y[0]) # mu_prior0 = mu0, V_prior0 = Q0
        
    _, (mu_1T, mu_prior_1T, V_1T, V_prior_1T, norm_fact_1T) = lax.scan(step, init=(mu_0, V_0), xs=(u[:-1], y[1:]))
        
    mu = jnp.concatenate([mu_0[jnp.newaxis, :], mu_1T])
    mu_prior = jnp.concatenate([mu0[jnp.newaxis, :], mu_prior_1T])
    V = jnp.concatenate([V_0[jnp.newaxis, :], V_1T])
    V_prior = jnp.concatenate([Q0[jnp.newaxis, :], V_prior_1T])
        
    # marginal log likelihood p(y_{1:T})
    ll = jnp.sum(norm_fact_1T) + norm_fact_0
    ll -= 0.5 * T * D * jnp.log(2 * jnp.pi)

    return mu, mu_prior, V, V_prior, ll

# jax efficient parallelization across batches
Kalman_filter_E_step_batches = jit(vmap(Kalman_filter_E_step, in_axes=(0, 0, None, None, None, None, None, None, None, None)))

def Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior):
    ''' 
    Kalman smoother in E-step that computes backwards: p(x_t | y_1:T) = Gaussian(m[t],cov[t])
            
    for each trial/session individually
    '''
    T = mu.shape[0]

    # last step (equal to last Kalman filter output)
    m_final = mu[-1]
    cov_final = V[-1]
        
    def step(state_next, args): # backward step in smoothing
        m_next, cov_next = state_next
        mu_current, mu_prior_next, V_current, V_prior_next = args
        
        # smoothing update
        L = jnp.linalg.solve(V_prior_next.T, A @ V_current.T).T
        m_current = mu_current + L @ (m_next - mu_prior_next)
        cov_current = V_current + L @ (cov_next - V_prior_next) @ L.T
        cov_successor_current = L @ cov_next
        return (m_current, cov_current), (m_current,cov_current,cov_successor_current)
    
    # gotta trick scan to go backwards in time by reversing inputs
    _, (m_0T1, cov_0T1, cov_successive) = lax.scan(step, init=(mu[-1], V[-1]), xs=(mu[:-1][::-1], mu_prior[1:][::-1], V[:-1][::-1], V_prior[1:][::-1]))
    
    # reversing back to chronological order
    m_0T1 = m_0T1[::-1]
    cov_0T1 = cov_0T1[::-1]
    cov_successive = cov_successive[::-1]
    
    m = jnp.concatenate([m_0T1, m_final[jnp.newaxis, :]])
    cov = jnp.concatenate([cov_0T1, cov_final[jnp.newaxis, :]])

    return m, cov, cov_successive

# jax efficient parallelization across batches
Kalman_smoother_E_step_batches = jit(vmap(Kalman_smoother_E_step, in_axes=(None, 0, 0, 0, 0)))

def sufficient_statistics_E_step(u, y, m, cov, cov_successive):
    ''' 
    compute auxillary matrices (sufficient statistics from E-step) to use in M-step updates
    
    for each session/trial individually
    '''

    M1 = jnp.sum(m, axis=0)
    M1_T = jnp.sum(cov[:-1], axis=0) + m[:-1].T @ m[:-1] # without last time point
    M_next = jnp.sum(cov_successive, axis=0) + m[:-1].T @ m[1:]
    Y1 = jnp.sum(y, axis=0)
    Y2 = y.T @ y # D x D sum of outer products y_t, y_t
    Y_tilde = m.T @ y # K x D sum of outer products m_t, y_t
    M_first = cov[0] + jnp.outer(m[0],m[0])
    M_last = cov[-1] + jnp.outer(m[-1],m[-1])
    U1_T = u[:-1].T @ u[:-1] # M x M sum of outer products u_t, u_t without last time point
    U_tilde = m[:-1].T @ u[:-1] # K x M sum of outer products m_t, u_t without last time point
    U_delta = m[1:].T @ u[:-1] # K x M sum of outer products m_t+1, u_t without last time point
        
    return M1, M1_T, M_next, Y1, Y2, Y_tilde, M_first, M_last, U1_T, U_tilde, U_delta

# jax efficient parallelization across batches
sufficient_statistics_E_step_batches = jit(vmap(sufficient_statistics_E_step, in_axes=(0,0,0,0,0)))

@partial(jit, static_argnums=(0,))
def closed_form_M_step(K1, u, y, A, B, Q, mu0, Q0, C, d, R, m, stats, verbosity):
    ''' 
    closed-form updates for all parameters except observation matrix C
        
    using all sessions together for updates
    '''

    S = y.shape[0]
    T = y.shape[1]
    D = C.shape[0]
    K = C.shape[1]
    M = B.shape[1]

    M1, M1_T, M_next, Y1, Y2, Y_tilde, M_first, M_last, U1_T, U_tilde, U_delta = stats
          
    # updates first latent (average over different trials/sessions S)
    mu0 = jnp.mean(m[:,0], axis=0)
    # Q0 = 1/S * (M_first - jnp.outer(jnp.sum(m[:,0], axis=0), mu0)- jnp.outer(mu0, jnp.sum(m[:,0], axis=0)) + S * jnp.outer(mu0,mu0.T))
    Q0 = (M_first - jnp.outer(jnp.sum(m[:,0], axis=0), mu0)- jnp.outer(mu0, jnp.sum(m[:,0], axis=0))) / S + jnp.outer(mu0,mu0.T)

    # update for d
    d = (Y1 - C @ M1) / (T*S)

    # update for R
    R = (Y2 + T * S * jnp.outer(d,d) - jnp.outer(d,Y1) - jnp.outer(Y1,d) - Y_tilde.T @ C.T - C @ Y_tilde + jnp.outer(d,M1) @ C.T + C @ jnp.outer(M1,d) + C @ M1_T @ C.T + C @ M_last @ C.T) / (T*S)
    R = 0.5 * (R + R.T) # to ensure numerical symmetry
    R += 1e-8 * jnp.eye(R.shape[0]) # to ensure no decay to 0 (ill conditioned)

    # # FOR NUMERICAL STABILITY, MIGHT HAVE TO USE scipy.linalg.solve INSTEAD OF np.linalg.inv
    
    # blockwise update for A
    Qinv = jnp.linalg.inv(Q)
    
    # update for A_11
    A11 = 2 * jnp.linalg.inv(Qinv[:K1,:K1]+Qinv[:K1,:K1].T) @ (Qinv[:K1,:K1].T @ M_next[:K1,:K1].T + Qinv[K1:,:K1].T @ M_next[:K1,K1:].T - 
                                0.5 * Qinv[K1:,:K1].T @ A[K1:,:K1] @ M1_T[:K1,:K1] - 0.5 * Qinv[:K1,K1:] @ A[K1:,:K1] @ M1_T[:K1,:K1] - 
                                0.5 * Qinv[:K1,K1:] @ A[K1:,K1:] @ M1_T[K1:,:K1] - 0.5 * Qinv[K1:,:K1].T @ A[K1:,K1:] @ M1_T[K1:,:K1] 
                                - Qinv[:K1,:K1].T @ B[:K1] @ U_tilde[:K1].T) @ jnp.linalg.inv(M1_T[:K1,:K1])
    # update for A_21
    A21 = 2 * jnp.linalg.inv(Qinv[K1:,K1:]+Qinv[K1:,K1:].T) @ (Qinv[:K1,K1:].T @ M_next[:K1,:K1].T + Qinv[K1:,K1:].T @ M_next[:K1,K1:].T - 
                                0.5 * Qinv[:K1,K1:].T @ A[:K1,:K1] @ M1_T[:K1,:K1] - 0.5 * Qinv[K1:,:K1] @ A[:K1,:K1] @ M1_T[:K1,:K1] - 
                                0.5 * Qinv[K1:,K1:] @ A[K1:,K1:] @ M1_T[K1:,:K1] - 0.5 * Qinv[K1:,K1:].T @ A[K1:,K1:] @ M1_T[K1:,:K1] 
                                - Qinv[:K1,K1:].T @ B[:K1] @ U_tilde[:K1].T) @ jnp.linalg.inv(M1_T[:K1,:K1])
    # update for A_22
    A22 = 2 * jnp.linalg.inv(Qinv[K1:,K1:]+Qinv[K1:,K1:].T) @ (Qinv[:K1,K1:].T @ M_next[K1:,:K1].T + Qinv[K1:,K1:].T @ M_next[K1:,K1:].T - 
                                0.5 * Qinv[:K1,K1:].T @ A[:K1,:K1] @ M1_T[:K1,K1:] - 0.5 * Qinv[K1:,:K1] @ A[:K1,:K1] @ M1_T[:K1,K1:] - 
                                0.5 * Qinv[K1:,K1:] @ A[K1:,:K1] @ M1_T[:K1,K1:] - 0.5 * Qinv[K1:,K1:].T @ A[K1:,:K1] @ M1_T[:K1,K1:] 
                                - Qinv[:K1,K1:].T @ B[:K1] @ U_tilde[K1:].T) @ jnp.linalg.inv(M1_T[K1:,K1:])
    
    A = jnp.block([[A11, jnp.zeros((K1,K-K1))],[A21, A22]])
        
    # blockwise update for B
    # U1_T += 1e-8 * np.eye(U1_T.shape[0]) # to avoid singular matrix
    # B = (U_delta - A @ U_tilde) @ np.linalg.inv(U1_T)
    B1 =  jnp.linalg.inv(Qinv[:K1,:K1]) @ (Qinv[:K1,:K1] @ U_delta[:K1] + Qinv[:K1,K1:] @ U_delta[K1:]
                        - Qinv[:K1,:K1] @ A[:K1,:K1] @ U_tilde[:K1] - Qinv[:K1,K1:] @ A[K1:,:K1] @ U_tilde[:K1]
                        - Qinv[:K1,K1:] @ A[K1:,K1:] @ U_tilde[K1:]) @ jnp.linalg.inv(U1_T)
    
    B = jnp.concatenate([B1, jnp.zeros((K-K1,M))])
    
    # update for Q
    Q = (M1_T - M_first + M_last + A @ M1_T @ A.T - A @ M_next - M_next.T @ A.T + B @ U1_T @ B.T - U_delta @ B.T - B @ U_delta.T + A @ U_tilde @ B.T + B @ U_tilde.T @ A.T) / ((T-1)*S)
    Q = 0.5 * (Q + Q.T) # to ensure numerical symmetry
    Q += 1e-8 * jnp.eye(Q.shape[0]) # to ensure no decay to 0
    
    return A, B, Q, mu0, Q0, d, R

def optimize_C_Stiefel(C, d, R, M1, M1_T, M_last, Y_tilde, max_iter_C=50, verbosity=0):
    ''' 
    closed-form updates for all parameters except the C
        
    using all sessions together for updates
    '''
    D = C.shape[0]
    K = C.shape[1]
    
    # inverse of R
    R_inv   = jnp.linalg.inv(R)

    # optimize over C
    manifold = Stiefel(n=D, p=K)
    @funjax(manifold)
    def loss_C(C_var): 
        ''' 
        optimize over Stiefel manifold of orthogonal matrices the loss function of C from EM
        '''
        # term1 = -trace(C Y_tilde R^{-1})  == -trace(Y_tilde @ (R^{-1} C))      
        term1 = -jnp.trace(C @ Y_tilde @ R_inv)                  # scalar

        # term2 = 0.5 * trace( (M1_T + M_last) @ (C^T R^{-1} C) ) 
        term2 = 0.5 * jnp.trace((M1_T + M_last) @ C_var.T @ R_inv @ C_var)              # scalar

        # term3 = (C M1) · (R^{-1} d)                                # (D,)
        term3 = jnp.dot(C_var @ M1, R_inv @ d)                     # scalar

        return term1 + term2 + term3
        
    problem = Problem(manifold, loss_C)
    optimizer = TrustRegions(max_iterations=max_iter_C, verbosity=verbosity)
    # optimizer = pymanopt.optimizers.ConjugateGradient(max_iterations=50, verbosity=verbosity) # too slow
    result = optimizer.run(problem, initial_point=C)
    C = result.point
    
    return C

def modified_M_step(K1, u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_successive, max_iter_C=50, verbosity=0):
    # per-session stats (each with leading S axis)
    sufficient_stats = sufficient_statistics_E_step_batches(u, y, m, cov, cov_successive)

    # sum across sessions for every leaf
    stats = tree_map(lambda x: x.sum(axis=0), sufficient_stats)
    
    M1, M1_T, _, _, _, Y_tilde, _, M_last, _, _, _ = stats

    C_new = optimize_C_Stiefel(C, d, R, M1=stats[0], M1_T=stats[1], M_last=stats[7], Y_tilde=stats[5], max_iter_C=max_iter_C, verbosity=verbosity)
    
    A_new, B_new, Q_new, mu0_new, Q0_new, d_new, R_new = closed_form_M_step(int(K1), u, y, A, B, Q, mu0, Q0, C, d, R, m, stats, verbosity)

    return A_new, B_new, Q_new, mu0_new, Q0_new, C_new, d_new, R_new 
