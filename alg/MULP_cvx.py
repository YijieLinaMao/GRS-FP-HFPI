
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
    # 全1
    weight=[]
    for i in range(number):
        weight.append(1)
    return weight

def Init_H_g(gamma,theta,N_user,Nt):
    h1=np.ones((Nt,1))-1j*np.zeros((Nt,1))
    for j in range(N_user-1):
        h=np.ones((Nt,1))-1j*np.zeros((Nt,1))
        for i in range(Nt):
            if i == 0:
                h[i, 0]=gamma[j]*h[i,0]
                continue
            h[i,0]=gamma[j]*np.cos(theta*(j+1)*i)-1j*gamma[j]*np.sin(theta*(j+1)*i)
        h1=np.concatenate((h1,h),axis=1)
    return h1



def P_init(P_p,H,N_user,alpha):
    P=[]
    for i in range(N_user):
        P.append(H[:,i][:,np.newaxis ]/np.linalg.norm(H[:,i])*np.sqrt(P_p*alpha[i]))
    return np.concatenate(P,axis=1)

def Init_P_g(H,N_user,Pt,Nt,alpha):
    P = P_init(Pt, H,N_user,alpha)
    return P

def count_T_MMSE_g(H,P,N_user):
    T=[]
    for k in range(N_user):
        # 计算该用户的私有T_k_k
        T_private_k=1
        for i in range(N_user):
            T_private_k+=abs_square_of_Hk_mul_Pi(H,P,k,i)
        T.append(T_private_k)
    return T

def count_I_MMSE_g(H,P,N_user):
    I=[]
    for k in range(N_user):
        I_k=1
        for i in range(N_user):
            if k==i:
                continue
            I_k+=abs_square_of_Hk_mul_Pi(H,P,k,i)
        I.append(I_k)
    return I

def count_g_MMSE_g(H,P,T,N_user):
    g=[]
    for k in range(N_user):
        g_k=count_g_i(Pi_mul_Hk(H,P,k,k),T[k])
        g.append(g_k)
    return g

def count_eplison_MMSE_g(T,I,N_user):
    eplison=[]
    for k in range(N_user):
        eplison_k=count_epsilon(T[k],I[k])
        eplison.append(eplison_k)
    return eplison

def count_u_MMSE_g(eplison,N_user):
    u=[]
    for k in range(N_user):
        u_k=count_u(eplison[k])
        u.append(u_k)
    return u

def count_ksai_g(u,eplison,N_user):
    ksai=[]
    for k in range(N_user):
        ksai_k=(count_ksai_i_cvx(eplison[k],u[k]))
        ksai.append(ksai_k)
    return ksai

def count_ksai_tot_g(ksai,N_user):
    ksai_tot=[]
    for k in range(N_user):
        ksai_tot_k=ksai[k]
        ksai_tot.append(ksai_tot_k)
    return ksai_tot


def count_eplison_cvx_g(T,g,H,P,N_user):
    eplison=[]
    for k in range(N_user):
        eplison_k=count_e_cvx(g[k],H,P,k,k,T[k])
        eplison.append(eplison_k)
    return eplison

def quad_form(P,phi,i,k):
    return cvx.quad_form(P[:, i][:, np.newaxis], phi[k])

def count_ksai(ksai,t,f,P,v,u,i,k):
    return ksai+t[k]-2*cvx.real(np.conj(f[k].T)@P[:,i][:, np.newaxis])-v[k]+u[k]

def count_ksai_imp(u,t,phi,f,v,P_opt,N_user):
    ksai=[0 for i in range(N_user)]
    for k in range(N_user):
        for i in range(N_user):
            ksai[k]+=quad_form(P_opt,phi,i,k)
    for k in range(N_user):
        ksai[k]=count_ksai(ksai[k],t,f,P_opt,v,u,k,k)
    return ksai

def optimize_P_x_g(u,t,phi,f,v,Pt,miu_weight_WSR,N_user,rth,Nt):
    P_opt = cvx.Variable((Nt, N_user),complex=True)
    ksai=count_ksai_imp(u,t,phi,f,v,P_opt,N_user)
    sum_P=0+0j
    for i in range(N_user):
        sum_P+=cvx.quad_form(P_opt[:,i][:, np.newaxis],np.eye(Nt))
    constraint2 = cvx.real(sum_P)<= Pt
    constraint3 = []
    for k in range(N_user):
        constraint3.append(ksai[k] <= 1-rth)
    constraints=[]
    constraints=constraints+constraint3
    constraints.append(constraint2)
    equation = 0
    for k in range(N_user):
        equation += miu_weight_WSR[k] * ksai[k]
    obj = cvx.Minimize(equation)
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    return P_opt.value,obj.value

def count_rate(T,I,N_user):
    rate=[]
    for k in range(N_user):
        rate.append(np.log2(T[k]/I[k]))
    return rate

def count_sum_rate(miu_weight_WSR,rate,N_user):
    sum_rate=0
    for i in range(N_user):
        sum_rate+=miu_weight_WSR[i]*rate[i]
    return sum_rate

def count_t(u,g,N_user):
    t=[]
    for k in range(N_user):
        t.append(u[k]*(abs(g[k])**2))
    return t

def count_phi(t,H,N_user):
    phi=[]
    H_square=[H[:, k][:, np.newaxis]@np.conj(H[:, k][:, np.newaxis].T) for k in range(N_user)]
    for k in range(N_user):
        phi.append(np.multiply(t[k],H_square[k]))
    return phi


def count_f(u,H,g,N_user):
    f=[]
    for k in range(N_user):
        f.append(u[k]*H[:, k][:, np.newaxis]*np.conj(g[k].T))
    return f

def count_v(u,N_user):
    v=[]
    for k in range(N_user):
        v.append(np.log2(u[k]))
    return v

def count_one_of_avg(H_real,P,N_user):
    T = count_T_MMSE_g(H_real, P, N_user)
    I = count_I_MMSE_g(H_real, P, N_user)
    e = count_eplison_MMSE_g(T, I, N_user)
    g = count_g_MMSE_g(H_real, P, T, N_user)
    u = count_u_MMSE_g(e, N_user)
    t=count_t(u, g, N_user)
    phi=count_phi(t, H_real, N_user)
    f=count_f(u, H_real, g, N_user)
    v=count_v(u, N_user)
    return u,t,phi,f,v

def count_avg(number_of_H_real,H_reals,P,N_user,Nt):
    u_sum,t_sum,phi_sum,v_sum,f_sum=[],[],[],[],[]
    for k in range(N_user):
        u_sum_i,t_sum_i, v_sum_i =0, 0, 0
        phi_sum_i = np.zeros((Nt, Nt)) + 1j * np.zeros((Nt, Nt))
        f_sum_i = np.zeros((Nt, 1))
        t_sum.append(t_sum_i)
        phi_sum.append(phi_sum_i)
        v_sum.append(v_sum_i)
        f_sum.append(f_sum_i)
        u_sum.append(u_sum_i)
    for i in range(len(H_reals)):
        H_real=H_reals[i]
        u,t,phi,f,v=count_one_of_avg(H_real, P, N_user)
        for k in range(N_user):
            u_sum[k]+=u[k]
            t_sum[k]+=t[k]
            phi_sum[k]+=phi[k]
            f_sum[k]=f[k]+f_sum[k]
            v_sum[k]+=v[k]
    for k in range(N_user):
        u_sum[k] =u_sum[k]/number_of_H_real
        t_sum[k]= t_sum[k]/number_of_H_real
        phi_sum[k] = phi_sum[k]/number_of_H_real
        f_sum[k] = f_sum[k]/number_of_H_real
        v_sum[k] = v_sum[k]/number_of_H_real
    return u_sum,t_sum,phi_sum,f_sum,v_sum

def count_avg_rate(number_of_H_real,H_reals,P,N_user):
    avg_rates=[0 for i in range(N_user)]
    for i in range(number_of_H_real):
        H_real=H_reals[i]
        T = count_T_MMSE_g(H_real, P, N_user)
        I = count_I_MMSE_g(H_real, P, N_user)
        rate = count_rate(T, I, N_user)
        avg_rates=[rate[i]+avg_rates[i] for i in range(N_user)]
    for k in range(N_user):
        avg_rates[k]=avg_rates[k]/number_of_H_real
    return avg_rates

def order_g(H_reals,H_estimate,miu_weight_WSR,boundary,Pt,N_users,Nt,rth,alpha):
    sum_rate=0
    sum = [0]
    L=[]
    P=Init_P_g(H_estimate,N_users,Pt,Nt,alpha)
    n = 0
    T1=time.time()
    number_of_H_real=len(H_reals)
    while True:
        n += 1
        u,t,phi,f,v=count_avg(number_of_H_real, H_reals, P, N_users, Nt)
        P,obj = optimize_P_x_g(u,t,phi,f,v,Pt,miu_weight_WSR,N_users,rth,Nt)
        sum.append(obj)
        if abs(sum[n] - sum[n - 1]) < boundary or n>=1000:
            rates=count_avg_rate(number_of_H_real, H_reals, P, N_users)
            sum_rate=count_sum_rate(miu_weight_WSR,rates,N_users)
            L.append(sum_rate)
            break
    T2=time.time()
    return sum_rate,T2-T1

def initial_H_random(seed, N_user, Nt,bias):
    H=[]
    np.random.seed(seed)
    for i in range(N_user):
        h=np.random.randn(Nt)+1j * np.random.randn(Nt)
        h=np.sqrt(bias[i])*h*1/np.sqrt(2)
        H.append(np.reshape(h,(Nt,1)))
    H=np.concatenate(H,axis=1)
    return H

if __name__ == "__main__":
    T1=time.time()
    Nt = 2
    N_users = 4
    rho = 0.1
    tolerance = 1e-4
    SNR=30
    Pt = 10 ** (SNR / 10)
    weight=[1,1,1,1]
    bias=[1,1,0.5,0.1]
    np.random.seed(5)
    H=initial_H_random(1, N_users, Nt, bias)
    alpha_i=0.6
    P_e = Pt ** (-alpha_i)
    H_estimate=initial_H_random(1, N_users, Nt, bias)*np.sqrt(1-P_e)
    H_errors=[]
    H_reals=[]
    for seed_i in range(3):
        H_error=initial_H_random(seed_i, N_users, Nt, bias)*np.sqrt(P_e)
        H_real=H_estimate+H_error
        H_errors.append(H_error)
        H_reals.append(H_real)
    SR,t=order_g(H_reals,H_estimate,weight, tolerance, Pt, N_users, Nt, 0, [0.25,0.25,0.25,0.25])
    print(SR)
    T2=time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))



