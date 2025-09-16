import matplotlib.pyplot as plt
from utils import *
import task_subspace_LDS
import jax.numpy as jnp

# def generate_random_color():
#   r = np.random.uniform(0, 1, 1)[0]
#   g = np.random.uniform(0, 1, 1)[0]
#   b = np.random.uniform(0, 1, 1)[0]
#   return (r, g, b)

# def plot_mse_parameters(axes, K1, A, B, Q, mu0, Q0, C, d, R, true_A, true_B, true_Q, true_mu0, true_Q0, true_C, true_d, true_R):
#     axes.set_ylabel('mse')
#     # axes.set_ylim(0,0.001)
#     axes.scatter(range(10), [mse(A[:K1,:K1], true_A[:K1,:K1]), mse(A[K1:,:K1], true_A[K1:,:K1]), mse(A[K1:,K1:], true_A[K1:,K1:]), mse(B[:K1], true_B[:K1]), mse(Q, true_Q), mse(mu0.reshape(mu0.shape[0],1), true_mu0.reshape(mu0.shape[0],1)), mse(Q0, true_Q0), mse(C, true_C), mse(d.reshape(d.shape[0],1), true_d.reshape(d.shape[0],1)), mse(R, true_R)])
#     axes.set_xticks(range(10), ['A11', 'A21', 'A22', 'B','Q', 'mu0', 'Q0', 'C_', 'd', 'R'])

def plot_eigenvalues(axes, eigval1, eigval2, color='black', label=['',''], alpha=1):
    axes.scatter(jnp.real(eigval1), jnp.imag(eigval1), color=color, label=label[0], marker='o', alpha=alpha)
    axes.scatter(jnp.real(eigval2), jnp.imag(eigval2), color=color, label=label[1], marker='s', alpha=alpha)
    axes.set_xlabel('Re(eigenvalue)')
    axes.set_ylabel('Im(eigenvalue)')
    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    axes.add_patch(circle1)
    axes.axvline(0, linestyle='dashed', color='black')
    axes.axhline(0, linestyle='dashed', color='black')