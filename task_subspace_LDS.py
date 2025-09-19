import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsci
import utils
from inference_and_sample import *

import time
from jax.tree_util import tree_map, tree_leaves

def _sync(x):
    # Ensure all pending device work is finished
    for leaf in tree_leaves(x):
        # Only DeviceArray leaves have block_until_ready
        try:
            leaf.block_until_ready()
        except AttributeError:
            pass
    return x

class task_subspace_LDS():
    """
    Class for two-coupled LDS
     
    """

    def __init__(self, D, K1, K2, M):
        ''' 
        D: int
            dimensionality of data y
        K1: int
            dimensionality of task-dynamics self-contained space (first LDS)
        K2: int
            dimensionality of the other space (second LDS)
        M: int
            dimensionality of inputs u
        '''
        self.K1 = K1
        self.K2 = K2
        self.K = K1 + K2 # latent dimensionality of both systems together
        self.D = D # dim of data
        self.M = M # dim of inputs

    def generate_dynamics_matrix(self, key, eigvals1, eigvals2, mean_N=0, var_N=1, disconnected = False):
        
        key11, key22, key21 = jr.split(key, num=3)

        # check that number of eigenvalues mathces latent dimensions
        if eigvals1.shape[0] != self.K1 or eigvals2.shape[0] != self.K2:
            raise Exception ('Number of eigenvalues in a system does not match its given dimensionality')

        # generating normal dynamics matrices for NOW !!!
        A1 = utils.generate_dynamics_A(key11, eigvals1, normal=True, distr='normal')
        A2 = utils.generate_dynamics_A(key22, eigvals2, normal=True, distr='normal')

        A = jnp.zeros((self.K, self.K))
        A = A.at[:self.K1,:self.K1].set(A1)
        A = A.at[self.K1:,self.K1:].set(A2)

        if disconnected == True:
            return A
        elif disconnected == False:
            A = A.at[self.K1:,:self.K1].set(mean_N + jr.normal(key21, (self.K2,self.K1)) * var_N ** 0.5)
            return A
    
    def generate_inputs(self, key, S, T, type='constant'):
        ''' 
        note that only system 1 directly receives inputs 
        '''
        # u = np.zeros((S,T,self.M))
        # cond1 = np.random.normal(0, 1, size=(self.M))
        # cond2 = np.random.normal(0, 1, size=(self.M))
        # u[:int(S/2),:] = cond1
        # u[int(S/2):,:] = cond2
        
        key1, key2 = jr.split(key, num=2)

        u = jnp.zeros((S,T,self.M))
        cond1 = jr.normal(key1, (1,T,self.M))
        cond2 = jr.normal(key2, (1,T,self.M))
        u = jnp.concatenate([jnp.repeat(cond1, int(S/2), axis=0), jnp.repeat(cond2, S - int(S/2), axis=0)])
        
        U1_T = utils.outer_prod_sum(u[:-1,:,:])
        
        if jnp.linalg.cond(U1_T) > 2 ** 15: # condition number too large and matrix not invertible
            raise Exception('U1_T is not invertible and will cause problems for updates of B')

        # if type == 'constant': # constant inputs 
        #     u[:,:] = np.random.normal(0, 1, size=(self.M))
        # else:
        #     raise Exception ('Need to include other options that constant inputs')
        
        return u

    def generate_other_parameters(self, key, A):
        ''' 

        generates parameters except dynamics matrix A and inputs u 
        Note that C is constrained to be orthonormal matrix 

        Parameters
        ----------
        D: int
            dimension of data y_t

        Returns
        -------
        w: N_weights x 1 numpy vector
            non-zero weight values

        s: int
            S = np.diag(s) is N x N covariance matrix of Gaussian RNN noise
        mu0: K x 1 numpy vector
            mean of Gaussian distr. of first latent
        Q0: K x K numpy array
            covariance of Gaussiant distr. of first latent
        C_: D x K numpy array
            output mapping from latents x_t to data y_t
        d: D x 1 numpy vector
            offset term for mapping of observations
        R: D x D numpy array
            covariance matrix of Gaussian observation noise
        '''
        
        # Q = np.random.normal(1, 0.2, (self.K, self.K))
        # Q = np.dot(Q, Q.T)
        # Q = 0.5 * (Q + Q.T)
        
        key_B, key_Q, key_C, key_d, key_R, key_Q0, key_mu0 = jr.split(key, num=7)

        # inputs only arrive in system 1
        B = jnp.concatenate([jr.normal(key_B, (self.K1,self.M)), jnp.zeros((self.K2,self.M))])

        # Q is diagonal for now
        Q = jnp.diag(jr.uniform(key_Q, (self.K,), minval=0.1, maxval=1.0))
        Q += 1e-8 * jnp.eye(Q.shape[0])
        
        # generate an orthonormal matrix C to actually be a projection matrix
        # C = jr.normal(key_C, (self.D,self.D))
        # C, _ = jnp.linalg.qr(C)  # QR decomposition, Q is the orthogonal matrix
        # C = C[:self.K,:].T
        C = jr.orthogonal(key_C, self.D)[:,:(self.K1+self.K2)]
        
        d = jr.normal(key_d, (self.D,)) + 2
        
        R = 0.1/jnp.sqrt(self.D) + 0.3/jnp.sqrt(self.D) * jr.normal(key_R, (self.D, self.D))
        R = R @ R.T # to make P.S.D
        R = 0.5 * (R + R.T) # to make symmetric
        R += 1e-8 * jnp.eye(R.shape[0])

        # Q0 = jsci.linalg.solve_discrete_lyapunov(A, Q)
        Q0 = jnp.diag(jr.uniform(key_Q0, (self.K,), minval=0.1, maxval=1.0))
        Q0 += 1e-8 * jnp.eye(Q0.shape[0])
        
        mu0 = jr.normal(key_mu0, (self.K,)) * 0.1 ** 2
        
        return  B, Q, mu0, Q0, C, d, R
    

    def fit_EM_timed(self, K1, u, y,
                     init_A, init_B, init_Q, init_mu0, init_Q0, init_C, init_d, init_R,
                     max_iter=300, verbosity=0, print_every=1):
        A  = jnp.asarray(init_A);   B  = jnp.asarray(init_B);   Q  = jnp.asarray(init_Q)
        mu0= jnp.asarray(init_mu0); Q0 = jnp.asarray(init_Q0)
        C  = jnp.asarray(init_C);   d  = jnp.asarray(init_d);   R  = jnp.asarray(init_R)

        S, T = y.shape[:2]
        
        # running once to warmup jit compilation
        mu, mu_prior, V, V_prior, _ = Kalman_filter_E_step_batches(y, u, A, B, Q, mu0, Q0, C, d, R)
        _sync((mu, mu_prior, V, V_prior))

        m, cov, cov_successive = Kalman_smoother_E_step_batches(A, mu, mu_prior, V, V_prior)
        _sync((m, cov, cov_successive))

        suff = sufficient_statistics_E_step_batches(u, y, m, cov, cov_successive)
        stats = tree_map(lambda x: x.sum(axis=0), suff)
        _sync(stats)

        _ = optimize_C_Stiefel(C, d, R,
                               M1=stats[0], M1_T=stats[1], M_last=stats[7], Y_tilde=stats[5],
                               max_iter_C=1, verbosity=0)

        A_, B_, Q_, mu0_, Q0_, d_, R_ = closed_form_M_step(int(K1), u, y, A, B, Q, mu0, Q0, C, d, R,
                                                           m, stats, verbosity)
        
        _sync((A_, B_, Q_, mu0_, Q0_, d_, R_))
        
        # timing from now on

        timing_log = []  

        for it in range(max_iter):
            t0 = time.perf_counter()
            mu, mu_prior, V, V_prior, ll = Kalman_filter_E_step_batches(y, u, A, B, Q, mu0, Q0, C, d, R)
            _sync((mu, mu_prior, V, V_prior, ll))
            t1 = time.perf_counter()

            m, cov, cov_successive = Kalman_smoother_E_step_batches(A, mu, mu_prior, V, V_prior)
            _sync((m, cov, cov_successive))
            t2 = time.perf_counter()

            suff = sufficient_statistics_E_step_batches(u, y, m, cov, cov_successive)
            _sync(suff)
            t3 = time.perf_counter()

            stats = tree_map(lambda x: x.sum(axis=0), suff)
            _sync(stats)
            t4 = time.perf_counter()

            # Time C optimization (Python loop; no _sync needed, but harmless)
            tC0 = time.perf_counter()
            C = optimize_C_Stiefel(C, d, R,
                                   M1=stats[0], M1_T=stats[1], M_last=stats[7], Y_tilde=stats[5],
                                   max_iter_C=10, verbosity=verbosity)
            tC1 = time.perf_counter()

            A, B, Q, mu0, Q0, d, R = closed_form_M_step(int(K1), u, y, A, B, Q, mu0, Q0, C, d, R,
                                                        m, stats, verbosity)
            _sync((A, B, Q, mu0, Q0, d, R))
            t5 = time.perf_counter()

            times = {
                "filter":       t1 - t0,
                "smoother":     t2 - t1,
                "suff_stats":   t3 - t2,
                "stats_reduce": t4 - t3,
                "opt_C":        tC1 - tC0,
                "closed_form":  t5 - tC1,
                "iteration":    t5 - t0,
            }
            timing_log.append(times)

            if (verbosity and (it % print_every == 0)):
                print(f"[it {it}] "
                      f"filter {times['filter']:.3f}s | smooth {times['smoother']:.3f}s | "
                      f"stats {times['suff_stats']:.3f}s | reduce {times['stats_reduce']:.3f}s | "
                      f"C {times['opt_C']:.3f}s | M {times['closed_form']:.3f}s | "
                      f"total {times['iteration']:.3f}s")

        return A, B, Q, mu0, Q0, C, d, R, timing_log

    
    
#     def fit_EM(self, u, y, init_A, init_B, init_Q, init_mu0, init_Q0, init_C, init_d, init_R, max_iter=300, verbosity=0):
#         # cast to jax arrays once (and to desired dtype)
#         A  = jnp.asarray(init_A, dtype=);  B  = jnp.asarray(init_B);  Q  = jnp.asarray(init_Q)
#         mu0= jnp.asarray(init_mu0);Q0 = jnp.asarray(init_Q0)
#         C  = jnp.asarray(init_C);  d  = jnp.asarray(init_d);  R  = jnp.asarray(init_R)

#         S, T = y.shape[:2]

#         for it in range(max_iter):
#             if it % 10 == 0:
#                 print(it)

#             # E-step (batched over S)
#             mu, mu_prior, V, V_prior, _ = Kalman_filter_E_step_batches(y, u, A, B, Q, mu0, Q0, C, d, R)
#             m, cov, cov_successive = Kalman_smoother_E_step_batches(A, mu, mu_prior, V, V_prior)

#             # M-step
#             A, B, Q, mu0, Q0, C, d, R = modified_M_step(self.K1, u, y, A, B, Q, mu0, Q0, C, d, R,
#                                                         m, cov, cov_successive, max_iter_C=50, verbosity=verbosity)

#         return A, B, Q, mu0, Q0, C, d, R
    
   

    
        # ecll = np.zeros(max_iter + 1)
        # elbo = np.zeros(max_iter + 1)
        # ll   = np.zeros((max_iter + 1, S))      # per-trial ll (kept for inspection)
        # ll_total = np.zeros(max_iter + 1)       # per-iteration total ll

        # for it in range(max_iter):
        #     if it % 10 == 0:
        #         print(it)

        #     # E-step (current θ): run filter/smoother for every trial
        #     m = np.zeros((S, T, self.K))
        #     cov = np.zeros((S, T, self.K, self.K))
        #     cov_next = np.zeros((S, T-1, self.K, self.K))

        #     for s in range(S):
        #         mu, mu_prior, V, V_prior, ll[it, s] = self.Kalman_filter_E_step(
        #             y[s], u[s], A, B, Q, mu0, Q0, C, d, R
        #         )
        #         m[s], cov[s], cov_next[s] = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)

        #     # totals for this θ (BEFORE M-step)
        #     ll_total[it] = ll[it].sum()

        #     # ECLL for this same θ
        #     ecll[it], _ = self.compute_ECLL(u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next)

        #     # Exact E-step ⇒ ELBO must equal marginal LL. Use identity to avoid entropy bugs.
        #     elbo[it] = ll_total[it]

        #     # (Optional: assert the equality numerically to catch issues early)
        #     # diff = ll_total[it] - ecll[it]
        #     # if np.abs(diff) > 1e-3:
        #     #     print(f"[warn] |LL - ECLL| = {diff:.3e} (expected to be entropy)")

        #     # M-step: update θ
        #     A, B, Q, mu0, Q0, C, d, R = self.modified_M_step(
        #         u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next, verbosity=verbosity
        #     )

        # # Final E-step after last update to populate the (+1)-th slot
        # m = np.zeros((S, T, self.K))
        # cov = np.zeros((S, T, self.K, self.K))
        # cov_next = np.zeros((S, T-1, self.K, self.K))
        # for s in range(S):
        #     mu, mu_prior, V, V_prior, ll[-1, s] = self.Kalman_filter_E_step(
        #         y[s], u[s], A, B, Q, mu0, Q0, C, d, R
        #     )
        #     m[s], cov[s], cov_next[s] = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)

        # ll_total[-1] = ll[-1].sum()
        # ecll[-1], _ = self.compute_ECLL(u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next)
        # elbo[-1] = ll_total[-1]

        # return ecll, elbo, ll_total, A, B, Q, mu0, Q0, C, d, R


        