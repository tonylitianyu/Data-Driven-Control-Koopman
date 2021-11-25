import autograd.numpy as np
from autograd import grad, jacobian
import time
import scipy
import typing

class StableKoopman():
    def __init__(self):
        ...

    ###utitlities
    def adjusted_frobenius_norm(self, X):
        return np.linalg.norm(X)**2/2

    def adjusted_modulo(self, x, div=100):
        result = (x % 100) 
        return result + (not result) * 100

    def get_max_abs_eigval(self,
        X: np.ndarray, 
        is_symmetric: bool = False) -> float:
        """
        Get maximum value of the absolute eigenvalues of
        a matrix X. Returns `numpy.inf` if the eigenvalue
        computation does not converge. X does not have to 
        be symmetric.
        """
        eigval_operator = (
            np.linalg.eigvalsh 
            if is_symmetric 
            else np.linalg.eigvals)

        try: 
            eig_max = max(abs(eigval_operator(X)))
        except:
            eig_max = np.inf
            
        return eig_max

    def project_psd(self,
        Q: np.ndarray, 
        eps: float = 0, 
        delta: float = np.inf) -> np.ndarray:
        """
        DEBUG
        """
        Q = (Q + Q.T)/2
        E,V = np.linalg.eig(a=Q)
        E_diag = np.diag(v=np.minimum(delta, np.maximum(E, eps)))
        Q_psd = V @ E_diag @ V.T
        return Q_psd

    def project_invertible(self, M, eps):
        """
        DEBUG
        """
        S,V,D = np.linalg.svd(M)
        if np.min(V) < eps:
            M = S @ np.diag(np.maximum(np.diag(V), eps)) @ D.T
        return M

    ###soc
    def checkdstable(self,
        A : np.ndarray,
        stab_relax: bool) -> typing.Tuple[np.ndarray, ...]:
        
        P = scipy.linalg.solve_discrete_lyapunov(a=A.T, q=np.identity(len(A)))
        S = scipy.linalg.sqrtm(A=P)
        S_inv = np.linalg.inv(a=S)
        OC = S @ A @ S_inv
        O,C = scipy.linalg.polar(a=OC, side='right')
        C = self.project_psd(Q=C, eps=0, delta=1-stab_relax)
        return P, S, O, C

    def initialize_soc_with_inputs(self, X,Y, U,**kwargs):
        """
        DEBUG
        """

        stability_relaxation = kwargs.get('stability_relaxation', 0)

        n = len(X)
        S = np.identity(n)
        S_inv = S
        XU = np.concatenate((X, U), axis=0)
        AB_ls = Y @ np.linalg.pinv(XU)

        O,C = scipy.linalg.polar(a=AB_ls[:n,:n], side='right')
        eig_max = self.get_max_abs_eigval(S_inv @ O @ C @ S, is_symmetric=False)
        B = AB_ls[:n,n:]
        
        if eig_max > 1 - stability_relaxation:
            C = self.project_psd(Q=C, eps=0, delta=1-stability_relaxation)
            e_old = self.adjusted_frobenius_norm(
                X=Y - (S_inv @ O @ C @ S) @ X - B @ U)
        else:
            e_old = self.adjusted_frobenius_norm(X=Y - AB_ls @ XU) 
        
        eig_max = max(1, eig_max)
        A_stab = AB_ls[:n,:n] / eig_max
        _,S_,O_,C_ = self.checkdstable(
            A=0.9999*A_stab, 
            stab_relax=stability_relaxation)
        S_inv_ = np.linalg.inv(a=S_)
        
        e_temp = self.adjusted_frobenius_norm(
            X=Y - (S_inv_ @ O_ @ C_ @ S_) @ X - B @ U)
        if e_temp < e_old:
            S, O, C = S_, O_, C_
            e_old = e_temp
        
        return e_old, S, O, C, B

    def get_gradients_with_inputs(self,
        X: np.ndarray, 
        Y: np.ndarray, 
        U: np.ndarray,
        B: np.ndarray,
        S: np.ndarray, 
        O: np.ndarray, 
        C: np.ndarray) -> typing.Tuple[np.ndarray, ...]:
        """
        Perform one gradient update step
        """

        S_inv = np.linalg.inv(a=S)
        M_ = S_inv @ O @ C @ S 
        N_ = - (X @ Y.T - X @ X.T @ M_.T - X @ U.T @ B.T) @ S_inv
        
        S_grad = (N_ @ O @ C - M_ @ N_).T 
        O_grad = (C @ S @ N_).T 
        C_grad = (S @ N_ @ O).T 

        B_grad = M_ @ X @ U.T + B @ U @ U.T - Y @ U.T
        
        return S_grad, O_grad, C_grad, B_grad

    def project_to_feasible(self,
        S: np.ndarray, 
        O: np.ndarray, 
        C: np.ndarray, 
        avoid_unstable: bool = False,
        **kwargs) -> typing.Tuple[np.ndarray, ...]:
        """
        Brief explanation
        """

        eps = kwargs.get('eps', 1e-12)
        stability_relaxation = kwargs.get('stability_relaxation', 0)
        
        S = self.project_invertible(M=S, eps=eps)

        matrix = O @ C
        if avoid_unstable:
            S_inv = np.linalg.inv(a=S)
            matrix = S_inv @ matrix @ S

        eig_max = self.get_max_abs_eigval(matrix, is_symmetric=False)
        if eig_max > 1 - stability_relaxation:
            O, _ = scipy.linalg.polar(a=O, side='right')
            C = self.project_psd(Q=C, eps=0, delta=1-stability_relaxation)
        return S, O, C

    def learn_stable_soc_with_inputs(self, X,Y, U, **kwargs):
        """
        DEBUG:
        - Make sure we test this for different dimensions of X and U
        """

        time_limit = kwargs.get('time_limit', 1800)
        alpha = kwargs.get('alpha', 0.5)
        step_size_factor = kwargs.get('step_size_factor', 5)
        fgm_max_iter = kwargs.get('fgm_max_iter', 20)
        conjugate_gradient = kwargs.get('conjugate_gradient', False)
        log_memory = kwargs.get('log_memory', True)
        eps = kwargs.get('eps', 1e-12)


        e100 = [None] * 100
        
        XU = np.concatenate((X, U), axis=0)
        AB_ls = Y @ np.linalg.pinv(XU)
        nA2 = self.adjusted_frobenius_norm(X=Y - AB_ls @ XU)
        e_old, S, O, C, B = self.initialize_soc_with_inputs(X,Y, U,**kwargs)


        Ys, Yo, Yc, Yb = S, O, C, B
        
        i = 1
        restart_i = True
        t_0 = time.time()
        while i:
            t_n = time.time()
            if t_n - t_0 > 1800:
                break
            alpha_prev = alpha

            S_grad, O_grad, C_grad, B_grad = self.get_gradients_with_inputs(X, Y, U, B, S, O, C)
            
            step = kwargs.get('step', 100)
            inneriter = 1
            
            Sn = Ys - S_grad * step
            On = Yo - O_grad * step
            Cn = Yc - C_grad * step
            Bn = Yb - B_grad * step
            
            Sn, On, Cn = self.project_to_feasible(Sn, On, Cn, avoid_unstable=True, **kwargs)
            
            Sn_inv = np.linalg.inv(a=Sn)
            e_new = self.adjusted_frobenius_norm(
                X=Y - (Sn_inv @ On @ Cn @ Sn) @ X - Bn @ U)


            while e_new > e_old and inneriter <= fgm_max_iter:
                
                Sn = Ys - S_grad * step
                On = Yo - O_grad * step
                Cn = Yc - C_grad * step
                Bn = Yb - B_grad * step
                
                Sn, On, Cn = self.project_to_feasible(Sn, On, Cn, avoid_unstable=True, **kwargs)
            
                Sn_inv = np.linalg.inv(a=Sn)
                e_new = self.adjusted_frobenius_norm(
                    X=Y - (Sn_inv @ On @ Cn @ Sn) @ X - Bn @ U)

                try:
                    assert (e_new < e_old) and (e_new > prev_error)
                    break
                except (AssertionError, NameError):
                        pass

                if inneriter == 1:
                    prev_error = e_new
                else:
                    if e_new < e_old and e_new > prev_error:
                        break
                    else:
                        prev_error = e_new
                step /= step_size_factor
                inneriter += 1

            alpha = (np.sqrt(alpha_prev**4 + 4*alpha_prev**2) - alpha_prev**2)/2
            beta = alpha_prev * (1 - alpha_prev) / (alpha_prev**2 + alpha)
            
            if e_new > e_old:
                if restart_i:
                    restart_i = False
                    alpha = kwargs.get('alpha', 0.5)
                    Ys, Yo, Yc, Yb, e_new = S, O, C, B, e_old
                else:
                    e_new = e_old
                    break
            else:
                restart_i = True
                if conjugate_gradient:
                    beta = 0
                Ys = Sn + beta * (Sn - S)
                Yo = On + beta * (On - O)
                Yc = Cn + beta * (Cn - C)
                Yb = Bn + beta * (Bn - B)
                
                S, O, C, B = Sn, On, Cn, Bn
            i+=1

            current_i = self.adjusted_modulo(i, 100)
            e100[current_i - 1] = e_old
            current_i_min_100 = self.adjusted_modulo(current_i + 1, 100)
            if (
                (e_old < 1e-6 * nA2) or 
                (
                    (i > 100) and 
                    (e100[current_i_min_100] - e100[current_i]) < 1e-8*e100[current_i_min_100]  
                    )
                ):
                break
            e_old = e_new


        # refine solutions
        A, B= self.refine_soc_inputs_solution(X, Y, U, B, S, O, C, **kwargs)
        
        # record memory usage
        mem = None
        if log_memory:
            object_mems = [value.nbytes for _, value in locals().items() if isinstance(value, np.ndarray)]
            mbs_used = sum(object_mems)/1e6
            mem = round(mbs_used, 3)

        return A, B, mem

    def refine_soc_inputs_solution(self, X, Y, U, B, S, O, C, **kwargs):
        """
        DEBUG
        - `e_0` and `delta` must become command-line arguments
        """
        
        e_0 = kwargs.get('e_0', 0.00001)
        delta = kwargs.get('delta', 0.00001)
        
        
        e_t = e_0
        S_inv = np.linalg.inv(a=S)
        A = S_inv @ O @ C @ S  
        n = len(A)

        XU = np.concatenate((X, U), axis=0)
        AB_ls = Y @ np.linalg.pinv(XU)

        A_B = np.concatenate((A, B), axis=1)
        grad = AB_ls - A_B

        AB_new = A_B + e_t * grad
        A_new = AB_new[:n,:n]
        
        # get initial max abs eigenvalue
        eig_max = self.get_max_abs_eigval(A_new, is_symmetric=False)
        
        while eig_max < 1 and self.adjusted_frobenius_norm(X=AB_new - AB_ls) > 0.01:
            e_t += delta
            AB_new = A_B + e_t * grad
            A_new = AB_new[:n,:n]
            eig_max = self.get_max_abs_eigval(A_new, is_symmetric=False)
        
        if e_t != e_0:
            A_Btemp = A_B + (e_t - delta) * grad
            A = A_Btemp[:n,:n]
            B = A_Btemp[:n,n:]
        
        return A, B