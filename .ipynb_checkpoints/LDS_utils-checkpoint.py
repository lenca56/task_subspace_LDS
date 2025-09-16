import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, lax
import jax
jax.config.update("jax_enable_x64", True)

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

def Kalman_filter_E_step(y, u, A, B, Q , mu0, Q0, C, d, R):
    ''' 
    for each trial/session individually

    note that inputs come in only at the second time step
    '''

    T = y.shape[0]
    K = A.shape[0]
    D = C.shape[0]
    
    CRC = C.T @ jnp.linalg.solve(R, C)  # C.T @ np.linalg.inv(R) @ C 
        
    def update_based_on_priors(mu_prior, V_prior, y_current):
            
        # normalizing factor
        norm_fact = - 0.5 * jnp.linalg.slogdet(C @ V_prior @ C.T + R)[1]
        norm_fact = norm_fact - 0.5 * (y_current - C @ mu_prior - d).T @ jnp.linalg.solve(C @ V_prior @ C.T + R, y_current - C @ mu_prior - d)
            
        # filter updates
        V_current = jnp.linalg.solve(CRC + jnp.linalg.solve(V_prior, jnp.eye(K)), jnp.eye(K)) # jnp.linalg.inv(CRC + jnp.linalg.inv(V_prior))
        CRy = C.T @ jnp.linalg.solve(R, y_current - d) # C.T @ np.linalg.inv(R) @ (y_current - d)
        mu_current = V_current @ (CRy + jnp.linalg.solve(V_prior, mu_prior)) # np.linalg.inv(V_prior_current) @ mu_prior_current)
            
        # to ensure precise symmetry
        V_current = 0.5 * (V_current + V_current.T)
            
        return mu_current, V_current, norm_fact
        
    def step(state_prev, args):
            
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
