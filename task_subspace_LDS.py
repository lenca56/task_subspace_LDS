import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsci
import jax
jax.config.update("jax_enable_x64", True)
import utils

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
        C = jr.orthogonal(key_C, self.D, (self.D,self.K))
        
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

    def generate_latents_and_observations(self, key, S, T, u, A, B, Q, mu0, Q0, C, d, R):
        ''' 
        Parameters
        ----------
        S: number of trials/sessions
        T: number of time points in trial/session
        '''

        x = np.zeros((S, T, self.K))
        y = np.zeros((S, T, self.D))

        for s in range(S):
            x[s, 0] = np.random.multivariate_normal(mu0.flatten(), Q0)
            y[s, 0] = np.random.multivariate_normal((C @ x[s, 0] + d).reshape(self.D), R)
            for i in range(1, T):
                x[s, i] = np.random.multivariate_normal((A @ x[s, i-1] + B @ u[s,i-1]).reshape((self.K)), Q)
                y[s, i] = np.random.multivariate_normal((C @ x[s, i] + d).reshape(self.D), R)
                
        return x, y

        