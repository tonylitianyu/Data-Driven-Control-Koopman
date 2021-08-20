import autograd.numpy as np
from autograd import grad, jacobian


class Koopman():
    def __init__(self, basis_f, num_basis, num_state):
        self.A = np.zeros((num_basis, num_basis))
        self.G = np.zeros((num_basis, num_basis))
        self.counter = 0

        self.basis = basis_f

        self.num_state = num_state



    def collect_data(self, state, new_state, action):
        phi_curr = self.basis(state, action)
        phi_next = self.basis(new_state, action)

        self.counter += 1
        self.A += np.outer(phi_next, phi_curr)/self.counter
        self.G += np.outer(phi_curr, phi_curr)/self.counter


    def get_full_K(self):
        return np.dot(self.A, np.linalg.pinv(self.G))


    def get_K_h_T(self):
        return self.get_full_K()[:self.num_state, :]

    