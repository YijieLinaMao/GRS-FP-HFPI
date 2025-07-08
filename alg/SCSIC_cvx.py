
import cvxpy as cvx
from pylab import *


def rec(m,n):
    return math.factorial(n)//(math.factorial(m)*math.factorial(n-m))

def count_set_number(n):
    sum=0
    for i in range(n):
        sum+=rec(i+1,n)
    return sum

def abs_square_of_Hk_mul_Pi(H,P,k,i):
    try:
        HP=abs(np.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])[0, 0])**2
    except:
        HP=abs_square_of_Hk_mul_Pi_cvx(H,P,k,i)
    return HP

def Pi_mul_Hk(H,P,k,i):
    return np.matmul(P[:, i].conjugate().T[np.newaxis, :], H[:, k][:, np.newaxis])[0, 0]

def count_g_i(abs_square_H_P,t):
    return abs_square_H_P*(t**(-1))

def count_epsilon(t,i):
    return (t**(-1))*i

def count_u(e):
    return e**(-1)

def count_ksai_i_cvx(e,ui):
    return ui*e-np.log2(ui)

def abs_square_of_Hk_mul_Pi_cvx(H,P,k,i):
    Hk_H = H[:, k].conjugate().T[np.newaxis, :]
    Pi = P[:, i][:, np.newaxis]
    return cvx.square(cvx.abs(cvx.matmul(Hk_H,Pi)))

def count_e_cvx(g,H,P,k,i,t):
    square_1=(abs(g))**2
    real_1=cvx.real(g*(cvx.matmul(H[:,k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])))
    e=square_1*t-2*real_1+1
    return e

def Init_weight_k(number):
    weight=[]
    for i in range(number):
        weight.append(1)
    return weight

def Init_H_g(gamma,theta,N_users,Nt):
    h1=np.ones((Nt,1))-1j*np.zeros((Nt,1))
    for j in range(N_users-1):
        h=np.ones((Nt,1))-1j*np.zeros((Nt,1))
        for i in range(Nt):
            if i == 0:
                h[i, 0]=gamma[j]*h[i,0]
                continue
            h[i,0]=gamma[j]*np.cos(theta*(j+1)*i)-1j*gamma[j]*np.sin(theta*(j+1)*i)
        h1=np.concatenate((h1,h),axis=1)
    return h1

def Init_P_g(H,Pt,N_users,alpha):
    P=[]
    power = Pt * alpha[0]
    a, b, c = np.linalg.svd(H)
    P.append(np.sqrt(power) * a[:, 0][:, np.newaxis])
    nums = [i for i in range(1, N_users)]
    for i in range(1,N_users-1):
        power = Pt * alpha[i]
        a, b, c = np.linalg.svd(H[:, nums])
        nums.pop(0)
        P.append(np.sqrt(power)*a[:,0][:,np.newaxis ])
    power = Pt * alpha[-1]
    P.append(H[:,-1][:,np.newaxis ]/np.linalg.norm(H[:,-1])*np.sqrt(power))
    return np.concatenate(P,axis=1)

def count_T_MMSE_g(H,P,N_users):
    T=[]
    for k in range(N_users):
        T_k=[]
        for i in range(k+1):
            T_k_i = 1
            for j in range(i,N_users):
                T_k_i+=abs_square_of_Hk_mul_Pi(H,P,k,j)
            T_k.append(T_k_i)
        T.append(T_k)
    return T

def count_I_MMSE_g(T,H,P,N_users):
    I=[]
    for k in range(N_users):
        I_k=T[k][1:len(T[k])]
        I_k.append(T[k][-1]-abs_square_of_Hk_mul_Pi(H,P,k,k))
        I.append(I_k)
    return I

def count_g_MMSE_g(H,P,T,N_users):
    g=[]
    for k in range(N_users):
        g_k=[]
        for i in range(len(T[k])):
            g_k.append(count_g_i(Pi_mul_Hk(H,P,k,i),T[k][i]))
        g.append(g_k)
    return g

def count_eplison_MMSE_g(T,I,N_users):
    eplison=[]
    for k in range(N_users):
        eplison_k=[]
        for i in range(len(T[k])):
            eplison_k.append(count_epsilon(T[k][i],I[k][i]))
        eplison.append(eplison_k)
    return eplison

def count_u_MMSE_g(eplison,N_users):
    u=[]
    for k in range(N_users):
        u_k=[]
        for i in range(len(eplison[k])):
            u_k.append(count_u(eplison[k][i]))
        u.append(u_k)
    return u

def count_ksai_g(u,eplison,N_users):
    ksai=[]
    for k in range(N_users):
        ksai_k=[]
        for i in range(len(eplison[k])):
            ksai_k.append(count_ksai_i_cvx(eplison[k][i],u[k][i]))
        ksai.append(ksai_k)
    return ksai

def count_WMMSEs(ksai,N_users):
    WMMSEs=[]
    for j in range(N_users):
        ksai_tot_k = []
        for k in range(j,N_users):
            ksai_tot_k.append(ksai[k][j])
        maxx=ksai_tot_k[0]
        for ksai_tot_i in ksai_tot_k:
            maxx=cvx.maximum(ksai_tot_i,maxx)
        WMMSEs.append(maxx)
    return WMMSEs


def count_eplison_cvx_g(T,g,H,P,N_users):
    eplison=[]
    for k in range(N_users):
        eplison_k = []
        for i in range(len(T[k])):
            eplison_k.append(count_e_cvx(g[k][i],H,P,k,i,T[k][i]))
        eplison.append(eplison_k)
    return eplison
def quad_form(P,phi,i,k,j):
    return cvx.quad_form(P[:, i][:, np.newaxis], phi[k][j])

def count_ksai(ksai,t,f,P,v,u,i,k,j):
    return ksai+t[k][j]-2*cvx.real(np.conj(f[k][j].T)@P[:,i][:, np.newaxis])-v[k][j]+u[k][j]

def count_ksai_imp(u,t,phi,f,v,P_opt,N_users):
    ksai=[]
    for k in range(N_users):
        item=len(u[k])
        ksai_k=[0 for i in range(item)]
        for i in range(k+1):
            for j in range(i,N_users):
                ksai_k[i]+=quad_form(P_opt,phi,j,k,i)
        for i in range(0, item):
            ksai_k[i] = count_ksai(ksai_k[i], t, f, P_opt, v, u, i, k,i)
        ksai.append(ksai_k)
    return ksai

def optimize_P_x_g(u,t,phi,f,v,Pt,miu_weight_WSR,N_users,Nt,rth):
    P_opt = cvx.Variable((Nt, N_users),complex=True)
    ksai = count_ksai_imp(u,t,phi,f,v,P_opt,N_users)
    WMMSEs= count_WMMSEs(ksai,N_users)
    sum_P=0+0j
    for i in range(N_users):
        sum_P+=cvx.quad_form(P_opt[:,i][:, np.newaxis],np.eye(Nt))
    constraint3 = []
    for k in range(N_users):
        constraint3.append(WMMSEs[k] <= 1-rth)
    constraint2 = cvx.real(sum_P)<= Pt
    constraints=[]
    constraints=constraints+constraint3
    constraints.append(constraint2)
    equation = 0
    for k in range(N_users):
        equation += miu_weight_WSR[k] * WMMSEs[k]
    obj = cvx.Minimize(equation)
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    return P_opt.value,obj.value

def count_rate(eplison,N_users):
    rate=[]
    for k in range(N_users):
        rate_k = []
        for i in range(len(eplison[k])):
            rate_k.append(-np.real(np.log2(eplison[k][i])))
        rate.append(rate_k)
    return rate

def count_rate_tot(rate,N_users):
    rates=[]
    for j in range(N_users):
        rates_k=[]
        for k in range(j,N_users):
            rates_k.append(rate[k][j])
        min=np.min(rates_k)
        rates.append(min)
    return rates

def count_sum_rate(miu_weight_WSR,rate_tot,N_users):
    sum_rate=0
    for i in range(N_users):
        sum_rate+=miu_weight_WSR[i]*rate_tot[i]
    return sum_rate

def count_avg_rate(number_of_H_real,H_reals,P,N_users):
    rates=[]
    rate=[]
    for i in range(number_of_H_real):
        H_real=H_reals[i]
        T = count_T_MMSE_g(H_real, P, N_users)
        I = count_I_MMSE_g(T, H_real, P, N_users)
        eplison=count_eplison_MMSE_g(T, I, N_users)
        rate=count_rate(eplison,N_users)
        rates.append(rate)
    avg_rates=[[0 for j in range(len(rate[i]))]for i in range(N_users)]
    for k in range(N_users):
        for i in range(len(avg_rates[k])):
            avg_rates[k][i]+=sum([rates[j][k][i] for j in range(number_of_H_real)])/number_of_H_real
    return avg_rates

def order_g(miu_weight_WSR,H_estimate,H_reals,boundary,Pt,N_users,Nt,rth,alphas):
    sum = [0]
    L=[]
    P=Init_P_g(H_estimate,Pt,N_users,alphas)
    n = 0
    T1=time.time()
    number_of_H_real=len(H_reals)
    while True:
        n += 1
        u,t,phi,f,v=count_avg(number_of_H_real,H_reals,P,N_users)
        P,obj = optimize_P_x_g(u,t,phi,f,v,Pt,miu_weight_WSR,N_users,Nt,rth)
        sum.append(obj)
        if abs(sum[n] - sum[n - 1]) < boundary or n>=1000:
            rates=count_avg_rate(number_of_H_real,H_reals,P,N_users)
            rate_tot=count_rate_tot(rates, N_users)
            sum_rate=count_sum_rate(miu_weight_WSR,rate_tot,N_users)
            L.append(sum_rate)
            break
    T2=time.time()
    return sum_rate,T2-T1

def count_t(u,g,N_users):
    t=[]
    for k in range(N_users):
        t_k=[]
        for i in range(len(u[k])):
            t_k.append(u[k][i]*(abs(g[k][i])**2))
        t.append(t_k)
    return t

def count_phi(t,H,N_users):
    phi=[]
    H_square=[H[:, k][:, np.newaxis]@np.conj(H[:, k][:, np.newaxis].T) for k in range(N_users)]
    for k in range(N_users):
        phi_k=[]
        for i in range(len(t[k])):
            phi_k.append(np.multiply(t[k][i],H_square[k]))
        phi.append(phi_k)
    return phi

def count_f(u,H,g,N_users):
    f=[]
    for k in range(N_users):
        f_k=[]
        for i in range(len(u[k])):
            f_k.append(u[k][i]*H[:, k][:, np.newaxis]*np.conj(g[k][i].T))
        f.append(f_k)
    return f

def count_v(u,N_users):
    v=[]
    for k in range(N_users):
        v_k=[]
        for i in range(len(u[k])):
            v_k.append(np.log2(u[k][i]))
        v.append(v_k)
    return v
def count_one_of_avg(H_real,P,N_users):
    T = count_T_MMSE_g(H_real, P, N_users)
    I = count_I_MMSE_g(T, H_real, P, N_users)
    e = count_eplison_MMSE_g(T, I, N_users)
    g = count_g_MMSE_g(H_real, P, T, N_users)
    u = count_u_MMSE_g(e, N_users)
    t=count_t(u, g, N_users)
    phi=count_phi(t, H_real, N_users)
    f=count_f(u, H_real, g, N_users)
    v=count_v(u, N_users)
    return u,t,phi,f,v

def count_avg(number_of_H_real,H_reals,P,N_users):
    u_sum,t_sum,phi_sum,f_sum,v_sum=count_one_of_avg(H_reals[0], P, N_users)
    for i in range(1,len(H_reals)):
        H_real=H_reals[i]
        u,t,phi,f,v=count_one_of_avg(H_real, P, N_users)
        for k in range(N_users):
            for j in range(len(u[k])):
                u_sum[k][j]+=u[k][j]
                t_sum[k][j]+=t[k][j]
                phi_sum[k][j]+=phi[k][j]
                f_sum[k][j]=f[k][j]+f_sum[k][j]
                v_sum[k][j]+=v[k][j]
    for k in range(N_users):
        for j in range(len(u_sum[k])):
            u_sum[k][j] =u_sum[k][j]/number_of_H_real
            t_sum[k][j] = t_sum[k][j]/number_of_H_real
            phi_sum[k][j] = phi_sum[k][j]/number_of_H_real
            f_sum[k][j] = f_sum[k][j]/number_of_H_real
            v_sum[k][j] = v_sum[k][j]/number_of_H_real
    return u_sum,t_sum,phi_sum,f_sum,v_sum

def initial_H_random(seed, N_users, Nt,bias):
    H=[]
    np.random.seed(seed)
    for i in range(N_users):
        h=np.random.randn(Nt)+1j * np.random.randn(Nt)
        h=np.sqrt(bias[i])*h*1/np.sqrt(2)
        H.append(np.reshape(h,(Nt,1)))
    H=np.concatenate(H,axis=1)
    return H
if __name__ == "__main__":
    T1=time.time()
    Nt = 2
    N_users = 3
    rho = 0.1
    tolerance = 1e-4
    SNR=30
    Pt = 10 ** (SNR / 10)
    weight=[1,1,1,1]
    alpha_i=1
    P_e = Pt ** (-alpha_i)
    bias=[1,0.5,0.1]
    H_estimate=initial_H_random(1, N_users, Nt, bias)*np.sqrt(1-P_e)
    H_errors=[]
    H_reals=[]
    for seed_i in range(3):
        H_error=initial_H_random(seed_i, N_users, Nt, bias)*np.sqrt(P_e)
        H_real=H_estimate+H_error
        H_errors.append(H_error)
        H_reals.append(H_real)
    alphas=[0.3, 0.3,0.4]
    SR,t = order_g(weight,  H_estimate,H_reals, tolerance, Pt, N_users, Nt, 0, alphas)
    print(SR)
    T2=time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))



