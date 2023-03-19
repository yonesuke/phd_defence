import numpy as np
from scipy import integrate
import sdeint

class VdP():
    def __init__(self):
        self.sigma = 0.01
        self.a1 = 0.3
        self.a2 = 0.7
        self.K1 = 0.01
        self.K2 = 0.01
        self.B = np.diag(self.sigma*np.ones(4))
        # seed
        self.seed = 1
    
    def G(self, x, t):
        """ noise (Input) """
        return  self.B 

    def F(self, x, t):
        u1, v1, u2, v2 = x 
        f_u1 = v1 + self.K1*(u2-u1) 
        f_v1 = self.a1*(1.0-u1**2)*v1 - u1 + self.K1*(u2**2)*v2
        f_u2 = v2 - self.K2*(u1**2)*v1
        f_v2 = self.a2*(1.0-u2**2)*v2 - u2 + self.K2*u1*(v1**2)
        return np.array([f_u1, f_v1, f_u2, f_v2])
    
    def solve_SDE(self, x0, tspan):
        np.random.seed(self.seed)
        return sdeint.itoint(self.F, self.G, x0, tspan)

class VdP_for_adjoint():
    def __init__(self):
        # 極大計算, 周期計算etc. の際に用いるconfig設定
        self.a = 0.7
        self.init_dt = 0.01 # 初期の刻み幅
        self.transient = 20000 # 収束させる時間
        self.grid_for_approximation = 2**3+1 # 極大近似計算において使用するgrid点の数
        self.num_bisection = 16 # 周期計算で使用する二分法回数
        self.rtol=1e-6
        self.atol=1e-9

    def F(self, t, x):
        u, v = x 
        f_u = v 
        f_v = self.a*(1.0-u**2)*v - u
        return np.array([f_u, f_v])

    def J(self, x):
        """ヤコビアン"""
        u, v = x
        return np.array([[0., 1.],
                         [-1.0-2.0*self.a*u*v, self.a*(1.0-u**2)]])

    def solve_ODE(self, tspan, x0):
        """ 軌道計算用 """
        sol = integrate.solve_ivp(self.F, tspan[[0, -1]], x0, t_eval=tspan)
        return sol
    

    def phase_zero_marker_and_period(self, init_x0=np.array([0.1, 0.1]), Poincare_v=0.0):
        """ ポアンカレ上の点を求め, 周期も求める."""
        def function(x0):
            """ x0計算して断面上の点とそれまでの時刻をだすルーチン"""
            # 時系列を一定時間計算する. (1周期を含むと仮定)
            dt = self.init_dt
            tspan = np.arange(0.0, 1000.0, dt)
            sol = integrate.solve_ivp(self.F, tspan[[0, -1]], x0, t_eval=tspan, 
                                        rtol=self.rtol, atol=self.atol)
            v = sol.y[1,:] # vのみの軌道
            X = sol.y.T # 全変数の軌道
            # 数値計算中でvがPoincare_vを負から正にまたぐ直前の状態点をとる.)
            arg = ((v[:-1] < Poincare_v) & (Poincare_v < v[1:]))
            v0, x0, t = v[:-1][arg][0], X[:-1][arg][0], tspan[:-1][arg][0]
            # 先程とった点から刻み幅を半分にして2step先まで計算し, 二分法探索によりポアンカレ断面の交点を探す. (交点はこの範囲に存在する.)
            for _ in range(self.num_bisection):
                dt *= 0.5
                tspan = np.array([0.0, dt, 2.0*dt])
                sol = integrate.solve_ivp(self.F, tspan[[0, -1]], x0, t_eval=tspan, 
                                            rtol=self.rtol, atol=self.atol)
                v = sol.y[1,:] # Vのみの軌道
                X = sol.y.T # 全変数の軌道
                # 真ん中の点が断面を超えている場合→進まない
                if (v[0] <= Poincare_v) & (v[1] > Poincare_v) & (v[2] >= Poincare_v):
                    v0, x0 = v[0], X[0] # ポアンカレ断面手前の点
                    v1, x1 = v[1], X[1] # ポアンカレ断面直後の点
                # 真ん中の点が断面を超えていない場合→dtだけ先に進む
                elif (v[0] <= Poincare_v) & (v[1] < Poincare_v) & (v[2] >= Poincare_v): 
                    v0, x0 = v[1], X[1] # ポアンカレ断面手前の点
                    v1, x1 = v[2], X[2] # ポアンカレ断面直後の点
                    t += dt
            
            # 最後に内分点を取り、断面上の点を取る, またその時刻も返す
            a0 = np.abs(v1 - Poincare_v) / (v1 - v0)
            a1 = np.abs(v0 - Poincare_v) / (v1 - v0)
            x0 = (a0*x0+a1*x1)/(a0+a1); x0[1] = Poincare_v
            t += (a1*dt)/(a0+a1)
            period = t
            return x0, period
            
        # 収束するまで計算
        dt = self.init_dt
        x0, tspan = init_x0, np.arange(0.0, self.transient, dt)  # x0 = [u0, v0]
        sol = integrate.solve_ivp(self.F, tspan[[0, -1]], x0, t_eval=tspan,
                                    rtol=self.rtol, atol=self.atol) 
        x0 = np.array(sol.y[:, -1])
        # 断面上の点を取る
        x0, _ = function(x0)
        # 再度断面まで戻ってくるまでの時間を計算
        _, period = function(x0) 
        return x0, period