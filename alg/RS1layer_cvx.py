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
    power = Pt * alpha[-1]
    a,b,c=np.linalg.svd(H)
    P.append(np.sqrt(power)*a[:,0][:,np.newaxis ])
    power_pri=Pt*alpha[0]
    alpha=[1/N_users for i in range(N_users)]
    for i in range(N_users):
        P.append(H[:,i][:,np.newaxis ]/np.linalg.norm(H[:,i])*np.sqrt(power_pri*alpha[i]))
    return np.concatenate(P,axis=1)

def count_T_MMSE_g(H,P,N_users):
    T=[]
    for k in range(N_users):
        T_k=[]
        T_private_k=1
        for i in range(N_users):
            T_private_k+=abs_square_of_Hk_mul_Pi(H,P,k,1+i)
        T_comm_k=T_private_k+abs_square_of_Hk_mul_Pi(H,P,k,0)
        T_k.append(T_comm_k)
        T_k.append(T_private_k)
        T.append(T_k)
    return T

def count_I_MMSE_g(T,H,P,N_users):
    I=[]
    for k in range(N_users):
        I_k=T[k][1:len(T[k])+1]
        I_private_k=1
        for i in range(N_users):
            if i !=k:
                I_private_k+=abs_square_of_Hk_mul_Pi(H,P,k,1+i)
        I_k.append(I_private_k)
        I.append(I_k)
    return I

def count_g_MMSE_g(H,P,T,N_users):
    g=[]
    for k in range(N_users):
        g_k=[]
        g_k.append(Pi_mul_Hk(H,P,k,0)*(T[k][0]**(-1)))
        g_k.append(count_g_i(Pi_mul_Hk(H,P,k,1+k),T[k][1]))
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

def count_ksai_tot_g(ksai,x,N_users):
    ksai_tot=[]
    for k in range(N_users):
        ksai_tot_k=ksai[k][-1]
        ksai_tot_k+=x[k][0]
        ksai_tot.append(ksai_tot_k)
    return ksai_tot

def count_eplison_cvx_g(T,g,H,P,N_users):
    eplison=[]
    for k in range(N_users):
        eplison_k=[]
        eplison_k.append(count_e_cvx(g[k][0],H,P,k,0,T[k][0]))
        eplison_k.append(count_e_cvx(g[k][-1],H,P,k,1+k,T[k][-1]))
        eplison.append(eplison_k)
    return eplison

def ksai_comm_constraint(x,ksai,N_users):
    constraints=[]
    x_comm_sum=1
    for k in range(N_users):
        x_comm_sum+=x[k][0]
    for i in range(N_users):
        constraints.append(ksai[i][0]<=x_comm_sum)
    return constraints


def optimize_P_x_g(u,g,H,Pt,miu_weight_WSR,N_users,Nt,rth):
    P_opt = cvx.Variable((Nt, N_users+1),complex=True)

    x=cvx.Variable((N_users,1))
    T = count_T_MMSE_g(H,P_opt,N_users)
    epsilon = count_eplison_cvx_g(T,g,H,P_opt,N_users)
    ksai = count_ksai_g(u,epsilon,N_users)
    ksai_tot = count_ksai_tot_g(ksai, x,N_users)

    sum_P=0+0j
    for i in range(N_users+1):
        sum_P+=cvx.quad_form(P_opt[:,i][:, np.newaxis],np.eye(Nt))
    constraint1 = ksai_comm_constraint(x,ksai,N_users)
    constraint2 = cvx.real(sum_P)<= Pt
    constraint3 = []
    for k in range(N_users):
        constraint3.append(ksai_tot[k] <= 1-rth)
    constraint4 = x<= np.zeros((N_users,1))
    constraints=[]
    constraints.append(constraint4)
    constraints=constraints+constraint3
    constraints.append(constraint2)
    constraints=constraints+constraint1
    equation = 0
    for k in range(N_users):
        equation += miu_weight_WSR[k] * ksai_tot[k]
    obj = cvx.Minimize(equation)
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.MOSEK)
    return P_opt.value,x.value,obj.value

def count_rate(T,I,N_users):
    rate=[]
    for k in range(N_users):
        rate.append(np.log2(T[k][-1]/I[k][-1]))
    return rate

def count_rate_tot(rate,x,N_users):
    rate_tot=[]
    for k in range(N_users):
        rate_tot_k=rate[k]
        rate_tot_k-=x[k][0]
        rate_tot.append(rate_tot_k)
    return rate_tot

def count_sum_rate(miu_weight_WSR,rate_tot,N_users):
    sum_rate=0
    for i in range(N_users):
        sum_rate+=miu_weight_WSR[i]*rate_tot[i]
    return sum_rate

def P_init(P_p,H ,N_users):
    P = []
    for i in range(N_users):
        P.append(H[:, i][:, np.newaxis] / np.linalg.norm(H[:, i]) * np.sqrt(P_p))
    return P

def P_total_init(Pt, H,N_users):
    P_c = Pt * 0.7
    P_p = (Pt - P_c) / N_users
    a, b, c = np.linalg.svd(H)
    p_c = np.sqrt(P_c) * a[:, 0][:, np.newaxis]
    P = P_init(P_p, H,N_users)
    P.append(p_c)
    return np.concatenate(P,axis=1)

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
            v_k.append(log2(u[k][i]))
        v.append(v_k)
    return v
def count_one_of_avg(H_real,P,N_users):
    T = count_T_MMSE_g(H_real,P,N_users)
    I = count_I_MMSE_g(T,H_real,P,N_users)
    e= count_eplison_MMSE_g(T,I,N_users)
    g = count_g_MMSE_g(H_real,P,T,N_users)
    u=count_u_MMSE_g(e,N_users)
    t=count_t(u, g, N_users)
    phi=count_phi(t, H_real, N_users)
    f=count_f(u, H_real, g, N_users)
    v=count_v(u, N_users)
    return u,t,phi,f,v

def count_avg(number_of_H_real,H_reals,P,N_users,Nt):
    u_sum,t_sum,phi_sum,v_sum,f_sum=[],[],[],[],[]
    for k in range(N_users):
        u_sum_k,t_sum_k, phi_sum_k,v_sum_k,f_sum_k = [], [], [], [],[]
        for j in range(2):
            u_sum_i,t_sum_i, v_sum_i =0, 0, 0
            phi_sum_i = np.zeros((Nt, Nt)) + 1j * np.zeros((Nt, Nt))
            f_sum_i = np.zeros((Nt, 1))
            t_sum_k.append(t_sum_i)
            phi_sum_k.append(phi_sum_i)
            v_sum_k.append(v_sum_i)
            f_sum_k.append(f_sum_i)
            u_sum_k.append(u_sum_i)
        t_sum.append(t_sum_k)
        phi_sum.append(phi_sum_k)
        v_sum.append(v_sum_k)
        f_sum.append(f_sum_k)
        u_sum.append(u_sum_k)
    for i in range(len(H_reals)):
        H_real=H_reals[i]
        u,t,phi,f,v=count_one_of_avg(H_real, P, N_users)
        for k in range(N_users):
            for j in range(2):
                u_sum[k][j]+=u[k][j]
                t_sum[k][j]+=t[k][j]
                phi_sum[k][j]+=phi[k][j]
                f_sum[k][j]=f[k][j]+f_sum[k][j]
                v_sum[k][j]+=v[k][j]
    for k in range(N_users):
        for j in range(2):
            u_sum[k][j] =u_sum[k][j]/number_of_H_real
            t_sum[k][j] = t_sum[k][j]/number_of_H_real
            phi_sum[k][j] = phi_sum[k][j]/number_of_H_real
            f_sum[k][j] = f_sum[k][j]/number_of_H_real
            v_sum[k][j] = v_sum[k][j]/number_of_H_real
    return u_sum,t_sum,phi_sum,f_sum,v_sum


def count_ksai(u,t,phi,f,v,P_opt,N_users):
    ksai=[]
    for k in range(N_users):
        ksai_c=0
        ksai_p=0
        for i in range(N_users):
            ksai_c+=cvx.quad_form(P_opt[:,i+1][:, np.newaxis],phi[k][0])
            ksai_p+=cvx.quad_form(P_opt[:,i+1][:, np.newaxis],phi[k][1])
        ksai_c+=cvx.quad_form(P_opt[:,0][:, np.newaxis],phi[k][0])
        ksai_p+=t[k][1]-2*cvx.real(np.conj(f[k][1].T)@P_opt[:,k+1][:, np.newaxis])-v[k][1]+u[k][1]
        ksai_c+=t[k][0]-2*cvx.real(np.conj(f[k][0].T)@P_opt[:,0][:, np.newaxis])-v[k][0]+u[k][0]
        ksai.append([ksai_c,ksai_p])
    return ksai

def optimize_P(u,t,phi,f,v,Pt,miu_weight_WSR,N_users,Nt,rth):
    P_opt = cvx.Variable((Nt, N_users+1),complex=True)
    x=cvx.Variable((N_users,1))
    ksai = count_ksai(u,t,phi,f,v,P_opt,N_users)
    ksai_tot = count_ksai_tot_g(ksai, x,N_users)
    sum_P=0+0j
    for i in range(N_users+1):
        sum_P+=cvx.quad_form(P_opt[:,i][:, np.newaxis],np.eye(Nt))
    constraint1 = ksai_comm_constraint(x,ksai,N_users)
    constraint2 = cvx.real(sum_P)<= Pt
    constraint3 = []
    for k in range(N_users):
        constraint3.append(ksai_tot[k] <= 1-rth)
    constraint4 = x<= np.zeros((N_users,1))
    constraints=[]
    constraints.append(constraint4)
    constraints=constraints+constraint3
    constraints.append(constraint2)
    constraints=constraints+constraint1
    equation = 0
    for k in range(N_users):
        equation += miu_weight_WSR[k] * ksai_tot[k]
    obj = cvx.Minimize(equation)
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.MOSEK)
    return P_opt.value,x.value,obj.value

def count_avg_rate(number_of_H_real,H_reals,P,N_users):
    rates=[]
    for i in range(number_of_H_real):
        H_real=H_reals[i]
        T=count_T_MMSE_g(H_real,P,N_users)
        I=count_I_MMSE_g(T,H_real,P,N_users)
        rate = count_rate(T, I, N_users)
        rates.append(rate)
    avg_rates=[0 for i in range(N_users)]
    for k in range(N_users):
        avg_rates[k]+=sum([rates[i][k] for i in range(number_of_H_real)])/number_of_H_real
    return avg_rates


import time
def order_g(miu_weight_WSR,H_estimate,H_reals,boundary,Pt,N_users,Nt,rth,alpha):
    sum = [0]
    L=[]
    P=Init_P_g(H_estimate,Pt,N_users,alpha)
    number_of_H_real=len(H_reals)
    n = 0
    T1=time.time()
    while True:
        n += 1
        u,t, phi, f, v=count_avg(number_of_H_real,H_reals,P,N_users,Nt)
        P,x,obj = optimize_P(u,t,phi,f,v,Pt,miu_weight_WSR,N_users,Nt,rth)
        sum.append(obj)
        if abs(sum[n] - sum[n - 1]) < boundary or n>=1000:
            avg_rate=count_avg_rate(number_of_H_real,H_reals,P,N_users)
            rate_tot=count_rate_tot(avg_rate, x, N_users)
            sum_rate=count_sum_rate(miu_weight_WSR,rate_tot,N_users)
            L.append(sum_rate)
            break
    T2=time.time()
    return sum_rate,T2-T1

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
    Nt = 4
    N_users = 3
    rho = 0.1
    tolerance = 1e-4
    SNR=30
    Pt = 10 ** (SNR / 10)
    weight=[1,1,1,1]
    alpha_i=0.8
    P_e = Pt ** (-alpha_i)
    bias=[1,0.5,0.1]
    H_estimate=initial_H_random(1, N_users, Nt, bias)*np.sqrt(1-P_e)
    H_errors=[]
    H_reals=[]
    for seed_i in range(1000):
        H_error=initial_H_random(seed_i, N_users, Nt, bias)*np.sqrt(P_e)
        H_real=H_estimate+H_error
        H_errors.append(H_error)
        H_reals.append(H_real)
    SR,t=order_g(weight, H_estimate,H_reals, tolerance, Pt, N_users, Nt, 0, [0.1,0.9])
    print(SR)
    print('程序运行时间:%s毫秒' % ((t) * 1000))




