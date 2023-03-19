from matplotlib.pyplot import grid
import numpy as np
import numpy.linalg as LA
from scipy.integrate import solve_ivp
from scipy.integrate import trapz

#
# Adjoint法の精度が悪い場合, self.gridを小さくするか,  
# 周期を求める精度を上げる(calculate_Xのrtol, atol), あるいはちゃんと周期軌道に収束しているか確認
#

class Adjoint():
    def __init__(self, F, J):
        self.F = F # F(t, x)
        self.J = J # J(x)
        self.grid = int(2**12+1) # Adjoint法の精度が悪い場合, ここを小さくするか, 周期を求める精度を上げる, あるいはちゃんと周期軌道に収束しているか確認
        self.eps1 = 10.0**-6  # calculate_X 精度確認用 (精度が悪くても止めない)
        self.eps2 = 10.0**-12  # calculate_Z 収束判定用


    def calculate_X(self, x0, period, rtol=1e-6, atol=1e-9):
        """  
        1周期の軌道 (時間順方向に計算すると収束. なお,Zは時間逆方向に計算すると収束) 
        F: F(t,x)
        x0: 周期軌道の内, 位相ゼロとする位置における状態点 (すでに既に周期軌道上に収束させたものとする)
        period: 事前に求めた, 周期軌道の周期
        self.eps1: 始点と終点の精度 (これ以下なら警告する.)
        """
        num_update = 0; recommended_grid = self.grid
        while period/(recommended_grid-1) > 0.01:
            num_update += 1
            recommended_grid = 2*(recommended_grid-1) + 1
        if num_update >= 1:
            print("Warning!!:\n self.grid should be changed, more than {:d} \n".\
                    format(recommended_grid,  period/(recommended_grid-1)),
                  flush=True)

        # 1周期を求める.
        tspan = np.linspace(0.0, period, num=self.grid)
        sol = solve_ivp(self.F, tspan[[0, -1]], x0, t_eval=tspan, 
                        rtol=rtol, atol=atol)
        start_x, end_x = x0, sol.y.T[-1]
        value = LA.norm(start_x-end_x)
        if value >= self.eps1:
            print("*** Warning! ***\n trajectory did not converge! \n<norm(x_start, x_end):{:0.4e}>".format(value))
        else:
            print("trajectory converged! \n<norm:{:0.4e}>".format(value))
        
        self.X, self.tspan = sol.y.T, sol.t 
        return np.copy(self.X), np.copy(self.tspan)
            

    def calculate_Z(self, X, tspan):
        """
        self.F: F(t, x)
        self.J: J(x)
        self.eps2: ZのZの収束判定に用いる.
        *** 以下はcalculate_X メソッドで得られたものを渡す ***. 
        X: 周期軌道
        tspan: Xの各行に対応する時間 
        """        
        self.X = X
        self.tspan = tspan

        def function1(z0):
            """ 1周期時間逆方向に PCR:Z を計算. 順方向に直して返す. """
            Z = np.zeros(self.X.shape)
            Z[-1] = z0
            
            # Heun
            for i in range(Z.shape[0]-1):
                dt = self.tspan[::-1][i+1] - self.tspan[::-1][i]
                
                x1 = self.X[::-1][i]; z1 = Z[::-1][i] 
                f1 = - self.J(x1).T @ z1 * dt
                
                x2 = self.X[::-1][i+1]; z2 = z1 + f1
                f2 = - self.J(x2).T @ z2 * dt
                Z[::-1][i+1] = z1 + (f1 + f2)/2.0
            
              # Euler (多分こっちのほうが適切かも)
#             for i in range(Z.shape[0]-1):
#                 dt = self.tspan[::-1][i+1] - self.tspan[::-1][i]
#                 x, z = self.X[::-1][i], Z[::-1][i]
#                 Z[::-1][i+1] = z - self.J(x).T @ z * dt
                
            return Z # 時間順方向で返す. 実際は0行目が最新の計算結果となる
        

        def function2(Z, period):
            """ 規格化 """
            dxdt = self.F
            omega = 2.0*np.pi/period
            normalized_Z = np.copy(Z)
            for i_1 in range(normalized_Z.shape[0]):
                x, z = self.X[i_1], normalized_Z[i_1]
                coef = omega / (z @ dxdt(None, x))
                z *= coef
            return normalized_Z

        # 周期
        period = self.tspan[-1]
        # 初期条件
        z0 = np.ones(self.X[0].size)
        # calculate first period
        new_Z = function1(z0)
        normalized_new_Z = function2(new_Z, period)
        
        # repetition
        num = int(0) # 更新回数
        while (1):
            # old <- new
            old_Z = np.copy(new_Z)
            normalized_old_Z = np.copy(normalized_new_Z)
            # 更新
            z0 = np.copy(normalized_old_Z[0])
            new_Z = function1(z0)
            normalized_new_Z = function2(new_Z, period)
            num += 1
            
            # 収束判定: 
            diff_array = LA.norm(new_Z - old_Z, axis=1)
            value = diff_array.max()
            if (value < self.eps2):
                print("convergence of Z: {:0.4e}".format(value))
                break
            if num % 5 == 0:
                print("convergence of Z: {:0.4e}".format(value))
            
            
        # 最後に規格化が働いているか確認.
        dxdt= self.F; omega = 2.0*np.pi/period
        diff = np.zeros([self.tspan.size,])
        for i_1 in range(self.tspan.size):
            x, z = self.X[i_1], normalized_new_Z[i_1]
            diff[i_1] = np.abs(z @ dxdt(None, x) - omega)
        print("max_t|Z(t)@dX/dt(t)-omega|={:0.4e}".format(diff.max()))
        
        self.Z = normalized_new_Z
        return np.copy(self.Z)

    
    
def Calculater_Gamma(coupling_function, phi, X_r, X_s, Z_r):
    """
    coupling_function: V(x_r, x_s) (function(x_r, x_s))
    phi:  
    X_r, X_s, Z_r: each row is correspondense to phi, i.e., x_r(phi)
    (X_r, X_s, Z_r is periodic)
    """
    func = coupling_function
    # r:receiver, s:sender
    phi_r = phi; 
    k = int(phi.size) 
    
    # PCRと相互作用の内積:ZV, 
    # row: phi_rの値に結びつけ, col: delta_phiの値に結びつけ 
    # (delta_phi = phi_s - phi_r)
    ZV = np.zeros([k, k])
    for i_1 in range(phi.size):   # i_1: phi_r のインデックス
        for i_2 in range(phi.size): # i_2: delta_phi のインデックス (phi_s = phi_r + delta_phi)
            # V(x_r, x_s), x_r=X_r(phi_r) coupling function
            x_r, x_s = X_r[i_1], X_s[(i_1+i_2)%(k-1)]
            z_r = Z_r[i_1]
            V_rs = coupling_function(x_r, x_s)
            ZV[i_1, i_2] = z_r @ V_rs
    # ZVをaxis=0の方向に積分,平均化すると位相結合関数となる. 
    # phi[-1] - phi[0] = 2pi
    Gamma = trapz(ZV, phi_r, axis=0) / (phi_r[-1]-phi_r[0])
    return Gamma
            
        
    
    
    




