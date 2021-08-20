import autograd.numpy as np
from autograd import grad, jacobian
import scipy
import scipy.linalg


class MPC():
    def __init__(self, horizon, K_h_T, basis_f, dt):
        #parameter
        self.horizon = horizon
        self.Q = np.diag([1.0,1.0,100.0,1.0]) 
        self.R = 100.0*np.eye(1) 
        self.RI = 1./self.R
        self.dldx = grad(self.loss_func, 0)
        self.K_h_T = K_h_T
        self.basis = basis_f
        self.dt = dt

        #init 
        self.u = np.zeros((horizon,1))
        self.dbasis_dx = jacobian(self.basis, 0)
        self.dbasis_du = jacobian(self.basis, 1)


    def forward(self, x_t, u_traj):
        
        curr_state = x_t.copy()
        traj = []
        loss = 0.0
        
        for t in range(self.horizon):

            traj.append(curr_state)
            
            curr_basis = self.basis(curr_state, u_traj[t]).reshape(-1,1)

            loss += self.loss_func(curr_state.reshape(-1,1), u_traj[t].reshape(-1,1))
            
            curr_state = (self.K_h_T @ curr_basis).flatten()  
            
        return traj, loss, curr_state
    
            
    def loss_func(self, xx, uu):
        return xx.T@self.Q@xx + uu.T@self.R@uu
    
    def backward(self, state_traj, u_traj):
        rho = np.array([0.0,0.0,0.0,0.0]).reshape(-1,1)
        result_u = np.zeros((self.horizon,1))
        
        for t in reversed(range(self.horizon)):
            curr_dldx = self.dldx(state_traj[t].reshape(-1,1), u_traj[t])
            
            curr_A_d = self.K_h_T@self.dbasis_dx(state_traj[t], u_traj[t])
            curr_B_d = self.K_h_T@self.dbasis_du(state_traj[t], u_traj[t])
            curr_A = (curr_A_d-np.eye(curr_A_d.shape[0]))/self.dt
            curr_B = curr_B_d/self.dt
            
            rho = rho - (- curr_dldx - curr_A.T@rho) * self.dt
            
            du = -self.RI@curr_B.T@rho
            
            result_u[t] = du[0]
            
        return result_u


    def __call__(self, state, init_step_size, beta, max_u):
        k = init_step_size
        state_traj, loss, last_state = self.forward(state, self.u)
        du_traj = self.backward(state_traj, self.u)

        temp_action_traj = self.u + du_traj * k

        _, J2u, _ = self.forward(state, temp_action_traj)
        
        
        last_J2u = loss
        while J2u < last_J2u:
            k = k * beta
            temp_action_traj = self.u + du_traj * k
            _, new_J2u, _ = self.forward(state, temp_action_traj)
            last_J2u = J2u
            J2u = new_J2u
            
        k = k / beta
        self.u = self.u + du_traj * k
        
        self.u = np.clip(self.u, -max_u, max_u) 
        return self.u[0]   #only apply the first action