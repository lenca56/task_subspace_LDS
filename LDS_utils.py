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
        u_prev, key_step = args
        key_x, key_y = jr.split(key_step, num=2)
        x_current = jr.multivariate_normal(key_x, A @ x_prev + B @ u_prev, Q)
        y_current = jr.multivariate_normal(key_y, C @ x_current + d, R)
        return x_current, (x_current, y_current)
        
    keys_step = jr.split(key_step, num = T - 1)
    _, (xs, ys) = lax.scan(step, init=x0, xs=(u[:-1], keys_step))
        
    x = jnp.concatenate([x0[jnp.newaxis, :],xs])
    y = jnp.concatenate([y0[jnp.newaxis, :],ys])
        
    return x, y