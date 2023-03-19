import matplotlib.pyplot as plt
import numpy as np
import math
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def b(x):
    return -np.cos(x)*(1-np.cos(x))

def critical_index(N):
    xs = np.array([2*np.pi*l/N for l in range(1,N)])
    ys = b(xs)
    sk = np.cumsum(ys)
    kc = np.argmax(sk>=0) + 1
    return kc

def additional_one(N,p):
    m = math.gcd(N, p)
    nt, pt = int(N/m), int(p/m)
    xs = np.array([2*np.pi*l/nt for l in range(1,nt)])
    ys = b(xs)
    sk = np.cumsum(ys)
    kc = np.argmax(sk>=0) + 1
    additional = 2*math.ceil(-m*sk[kc-2]/ys[kc-1]-1)
    return additional

def optimal_Np(N,p):
    m = math.gcd(N, p)
    nt, pt = int(N/m), int(p/m)
    if nt < 5:
        return "there is no feasible solution"
    kc = critical_index(nt)
    xs = np.zeros(N)
    # step 1
    for i in range(m):
        for j in range(1,kc):
            xs[i*nt+j] = 1.0
            xs[i*nt+nt-kc+j] = 1.0
    # step 2
    for i in range(1,m):
        xs[i*nt] = 1.0
    # step 3
    additional = additional_one(N, p)
    # first enumerate all the possible candidates
    candidates = []
    for i in range(m):
        candidates.append(i*nt+kc)
        candidates.append((i+1)*nt-kc)
    for i in range(int(additional/2)):
        # additionally select from left to right
        xs[candidates[i]] = 1.0
        # also, don't forget to satisfy 
        xs[N-candidates[i]] = 1.0
    return xs

def vec_aij_fft(thetas, ys):
    es = np.exp(1j*thetas)
    return np.imag(np.conjugate(es)*np.fft.ifft(np.fft.fft(es)*ys))

def distance_from_equilibrium(xs, x0):
    vec = np.mod(xs-x0, 2*np.pi)
    return np.linalg.norm(np.where(vec>np.pi, 2*np.pi-vec, vec))

def runge_kutta(x0, dt, tmax, v, args_v, rec, args_rec):
    '''
    solve ode with vector field `v` and return timeseries of values recorded by function `rec`
    ## Parameters
    `x0` : numpy array, initial value of the ODE
    `dt` : float, time step of Runge Kutta algorithm
    `tmax` : float, maximum computation time
    `v` : function(x, *args) (input: numpy array (unknown function x) and other argument (args) , output: numpy array), vector field of ODE
    `args_v` : list, arguments of `v` other than input unknown function
    `rec` : function(x, *args) (input: numpy array (unknown function x) and other argument (args), output: any), values you want to observe throughout the computation
    `args_rec` : list, arguments of `rec` other than input unknown function
    '''
    loops = int(tmax/dt)
    x_now = x0.copy()
    ans = []
    for _ in range(loops):
        k1 = v(x_now, *args_v)
        k2 = v(x_now + 0.5 * dt * k1, *args_v)
        k3 = v(x_now + 0.5 * dt * k2, *args_v)
        k4 = v(x_now + dt * k3, *args_v)
        x_next = x_now + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        ans.append(rec(x_next, *args_rec))
        x_now = x_next.copy()
    return np.array(ans)

m = 100
N, p = 19*m, m

# optimal network
xs = optimal_Np(N, p)
xs_fft = np.fft.fft(xs)
# maximum eigenvalue
lambda_p = 0
for i in range(1,N):
    lambda_p += xs[i]*np.cos(p*2.0*np.pi*i/N)*(-1.0+np.cos(p*2.0*np.pi*i/N))

list_distances = []

for _ in range(5):
    # initial condition
    # theta0: equilibrium point
    # thetas: initial phases
    # mean of noises is shifted to zero
    # in order to avoid global rotation
    theta0 = np.mod(np.array([2*np.pi*p*i/N for i in range(N)]), 2*np.pi)
    noises = np.random.normal(scale = 0.1*np.pi/np.sqrt(N), size = N)
    noises -= noises.mean()
    thetas = theta0 + noises

    dt, tmax = 0.001, 4
    distances = runge_kutta(thetas, dt, tmax, vec_aij_fft, [xs_fft], distance_from_equilibrium, [theta0])
    list_distances.append(distances)

#plot
plt.figure(figsize=(8, 6), tight_layout=True)
fig, ax1 = plt.subplots()
left, bottom, width, height = [0.4, 0.4, 0.4, 0.4]
ax2 = fig.add_axes([left, bottom, width, height])
plt.rcParams['font.size']=28
#ax1.rcParams['font.size']=28
#ax2.rcParams['font.size']=28
ts = np.arange(0, tmax, dt)
# ax1
ax1.set_xlim(-dt,tmax)
ax1.set_ylim(0,0.2)
for distances in list_distances:
    ax1.plot(ts, distances, color="tab:blue")
ax1.set_xlabel(r"$t$", fontsize=20)
ax1.set_ylabel(r"$\|\bm{\theta}(t)-\bm{\theta}_{p}^{\ast}\|$", fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)

# ax2
ax2.set_xlim(-dt, tmax)
for distances in list_distances:
    ax2.plot(ts,np.log(distances), color="tab:blue")
ax2.plot(ts,-5.5 + lambda_p*ts, linestyle="dashed",label="$\lambda_p t+\mathrm{const.}$", color="tab:orange")
ax2.legend(fontsize=15)
ax2.set_xlabel(r"$t$", fontsize=20)
ax2.set_ylabel(r"$\log\|\bm{\theta}(t)-\bm{\theta}_{p}^{\ast}\|$", fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)

plt.savefig("../figs/ode.pdf", bbox_inches="tight")