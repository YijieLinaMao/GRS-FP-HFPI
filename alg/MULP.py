import numpy as np
import copy
import time
import platform
def abs_square_of_Hk_mul_Pi(H,P,k,i):
    HP=abs(np.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])[0, 0])**2
    return HP
def square_of_Hk(H,k):
    H_square=np.matmul(H[:, k][:,np.newaxis], H[:, k].conjugate().T[ np.newaxis, :])
    return H_square
def Hk_mul_Pi(H,P,k,i):
    return np.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])[0,0]
def count_T(H,P,N_users):
    T_pri=[]
    for k in range(N_users):
        T_private_k=1
        for i in range(N_users):
            T_private_k+=abs_square_of_Hk_mul_Pi(H,P,k,i)
        T_pri.append(T_private_k)
    return T_pri
def count_I(T_pri,H,P,N_users):
    I_pri=[]
    for k in range(N_users):
        I_private_k=T_pri[k]-abs_square_of_Hk_mul_Pi(H,P,k,k)
        I_pri.append(I_private_k)
    return I_pri
def count_SINR(I_pri,N_users,P,H):
    SINR_comm,SINR_pri=[],[]
    for k in range(N_users):
        SINR_pri.append(abs_square_of_Hk_mul_Pi(H,P,k,k)/I_pri[k])
    return SINR_comm,SINR_pri

def count_beta(alpha_pri,N_users,P,T_pri,H):
    beta_pri=[]
    for k in range(N_users):
        beta_pri_k=np.sqrt(1+alpha_pri[k])*Hk_mul_Pi(H,P,k,k)/T_pri[k]
        beta_pri.append(beta_pri_k)
    return beta_pri

def count_mp(beta_pri,beta_comm,H,N_users):
    summ=0
    for k in range(N_users):
        summ+=(abs(beta_pri[k])**2*(abs(beta_comm[k])**2))*square_of_Hk(H,k)
    return summ

def count_first_part_of_update_P_pri(beta_pri,beta_comm,H,mu,N_users,Nt):
    return count_mp(beta_pri,beta_comm,H,N_users)+mu*np.eye(Nt)

def count_second_part_of_update_P_pri(alpha_pri,beta_pri,H,k):
    summ=np.sqrt(1+alpha_pri[k])*beta_pri[k]*H[:, k][:, np.newaxis]
    return summ

def update_P_pri(alpha_pri,beta_comm,beta_pri,H,N_users,Nt,mu):
    P=[]
    part_1 = count_first_part_of_update_P_pri(beta_pri, beta_comm, H, mu, N_users, Nt)
    part_1_inv=np.linalg.inv(part_1)
    for k in range(N_users):
        part_2 = count_second_part_of_update_P_pri(alpha_pri,beta_pri,H,k)
        P.append(part_1_inv@part_2)
    return np.concatenate(P,axis=1)


def count_g_pri(H,P,alpha_pri,beta_pri,N_users,T_pri):
    g=[]
    for k in range(N_users):
        g_part_1=np.log2(1+alpha_pri[k])-alpha_pri[k]
        g_part_2=2*np.sqrt(1+alpha_pri[k])*np.real(np.conj(beta_pri[k].T)*Hk_mul_Pi(H,P,k,k))
        g_part_3=(abs(beta_pri[k])**2)*T_pri[k]
        g.append(g_part_1+g_part_2-g_part_3)
    return g

def found_order(H,N_users):
    H_strength=[]
    diction=dict()
    for i in range(N_users):
        h=np.linalg.norm(H[:,i])
        H_strength.append(h)
        diction[h]=i
    H_strength2=copy.deepcopy(H_strength)
    sum_1=sum(H_strength)
    H_strength3=[]
    for h_strength in H_strength:
        H_strength3.append(h_strength/sum_1)
    H_strength.sort()
    H_strength.reverse()
    H_strength3.sort()
    herms=[]
    for h in H_strength:
        herms.append(diction[h])
    for j in herms:
        H_strength2[herms[j]]=H_strength3[j]
    return H_strength2

def count_e(mu,mu_new):
    e=0
    e += abs(mu- mu_new)
    return e

def count_rate(T,I,N_users):
    rate=[]
    for k in range(N_users):
        rate.append(np.log2(T[k]/I[k]))
    return rate

def count_np(alpha_pri, beta_pri, H,N_users):
    n_p=[]
    for k in range(N_users):
        n_p.append(count_second_part_of_update_P_pri(alpha_pri, beta_pri, H, k))
    return n_p

def count_alpha(T,I,N_users):
    return [T[k]/I[k]-1 for k in range(N_users)]

def count_avg_alpha_beta_one(H,P,N_users):
    T_pri = count_T(H, P, N_users)
    I_pri = count_I(T_pri, H, P, N_users)
    alpha_pri = count_alpha(T_pri, I_pri, N_users)
    beta_pri = count_beta( alpha_pri, N_users, P, T_pri, H)
    return alpha_pri,beta_pri

def count_avg_of_m(ms,numbers_of_H_real):
    m_sum=np.zeros_like(ms[0])
    for i in range(numbers_of_H_real):
        m_sum+=ms[i]
    return m_sum/numbers_of_H_real

def count_avg_of_alpha_beta(alphas,numbers_of_H_real,N_users):
    avg=[]
    for k in range(N_users):
        avg.append(0)
    for i in range(numbers_of_H_real):
        for k in range(N_users):
            avg[k]+=alphas[i][k]
    for k in range(N_users):
        avg[k]/=numbers_of_H_real
    return avg

def count_avg_of_np(alphas,numbers_of_H_real,N_users):
    avg=[]
    for k in range(N_users):
        avg.append(np.zeros_like(alphas[0][0]))
    for i in range(numbers_of_H_real):
        for k in range(N_users):
            avg[k]+=alphas[i][k]
    for k in range(N_users):
        avg[k]/=numbers_of_H_real
    return avg



def count_m_n_r_one(H_reals,numbers_of_H_real,P,N_users,Nt):
    m_p_s=[np.zeros((Nt,Nt)) for k in range(N_users)]
    n_p_s=[np.zeros((Nt,1)) for k in range(N_users)]
    r_p_s=[0 for k in range(N_users)]
    for i in range(numbers_of_H_real):
        H_real=H_reals[i]
        alpha_pri,beta_pri=count_avg_alpha_beta_one(H_real, P, N_users)
        beta_pri_square=[abs(beta_pri[k])**2 for k in range(N_users)]
        square_of_H=[square_of_Hk(H_real,k) for k in range(N_users)]
        m_p=[beta_pri_square[k]*square_of_H[k] for k in range(N_users)]
        n_p=[np.sqrt(1+alpha_pri[k])*beta_pri[k]*H_real[:, k][:, np.newaxis] for k in range(N_users)]
        r_p=[np.log2(1+alpha_pri[k])-beta_pri_square[k]-alpha_pri[k] for k in range(N_users)]
        m_p_s=[m_p_s[k]+m_p[k] for k in range(N_users)]
        n_p_s=[n_p_s[k]+n_p[k] for k in range(N_users)]
        r_p_s=[r_p_s[k]+r_p[k] for k in range(N_users)]
    m_p_avg = [m_p_s[k]/numbers_of_H_real for k in range(N_users)]
    n_p_avg= [n_p_s[k]/numbers_of_H_real for k in range(N_users)]
    r_p_avg = [r_p_s[k] /numbers_of_H_real for k in range(N_users)]
    return m_p_avg,n_p_avg,r_p_avg

def count_alpha_beta_avg(H_reals,numbers_of_H_real,P,N_users):
    alpha_comm_s, beta_comm_s,alpha_pri_s,beta_pri_s=[],[],[],[]
    beta_comm_square,beta_pri_square=[],[]
    for i in range(numbers_of_H_real):
        H_real=H_reals[i]
        alpha_pri,beta_pri=count_avg_alpha_beta_one(H_real, P, N_users)
        alpha_pri_s.append(alpha_pri)
        beta_pri_s.append(beta_pri)
        beta_pri_square.append([abs(beta_pri[i])**2for i in range (N_users)])
    alpha_pri_avg=count_avg_of_alpha_beta(alpha_pri_s, numbers_of_H_real, N_users)
    beta_pri_square_avg=count_avg_of_alpha_beta(beta_pri_square, numbers_of_H_real, N_users)
    return alpha_pri_s,beta_pri_s,alpha_pri_avg,beta_pri_square_avg

def count_m_n_one(alpha_comm_s,beta_comm_s,alpha_pri_s,beta_pri_s,i,H,N_users):
    m_p,n_p,m_c,n_c=[],[],[],[]
    for k in range(N_users):
        m_p.append(abs(beta_pri_s[i][k])**2*square_of_Hk(H,k))
        n_p.append(np.sqrt(1+alpha_pri_s[i][k])*beta_pri_s[i][k]*H[:, k][:, np.newaxis])
        m_c.append(abs(beta_comm_s[i][k])**2*square_of_Hk(H,k))
        n_c.append(np.sqrt(1+alpha_comm_s[i][k])*beta_comm_s[i][k]*H[:, k][:, np.newaxis])
    return m_p,n_p,m_c,n_c

def count_m_n_avg(alpha_comm_s,beta_comm_s,alpha_pri_s,beta_pri_s,H_reals,N_users,numbers_of_H_real):
    m_p_s,n_p_s,m_c_s,n_c_s=[],[],[],[]
    for i in range(numbers_of_H_real):
        H_real=H_reals[i]
        m_p, n_p, m_c, n_c=count_m_n_one(alpha_comm_s,beta_comm_s,alpha_pri_s,beta_pri_s,i,H_real,N_users)
        m_c_s.append(m_c)
        m_p_s.append(m_p)
        n_c_s.append(n_c)
        n_p_s.append(n_p)
    m_c_avg=count_avg_of_np(m_c_s,numbers_of_H_real,N_users)
    m_p_avg=count_avg_of_np(m_p_s,numbers_of_H_real,N_users)
    n_c_avg=count_avg_of_np(n_c_s,numbers_of_H_real,N_users)
    n_p_avg = count_avg_of_np(n_p_s,numbers_of_H_real,N_users)
    return m_c_avg,m_p_avg,n_c_avg,n_p_avg

def count_t_v_f_one(alpha_comm_s,beta_comm_s,alpha_pri_s,beta_pri_s,i,H,N_users):
    v_c,v_p,t_c,t_p,f_c,f_p=[],[],[],[],[],[]
    for k in range(N_users):
        v_c.append(np.log2(1+alpha_comm_s[i][k]))
        v_p.append(np.log2(1+alpha_pri_s[i][k]))
        f_p.append(2*np.sqrt(1+alpha_pri_s[i][k])*np.conj(beta_pri_s[i][k].T)*np.conj(H[:, k][:, np.newaxis].T))
        f_c.append(2*np.sqrt(1+alpha_comm_s[i][k])*np.conj(beta_comm_s[i][k].T)*np.conj(H[:, k][:, np.newaxis].T))
    return v_c,v_p,f_c,f_p

def count_t_v_f_avg(alpha_pri,alpha_comm,beta_pri,beta_comm,H_reals,N_users,numbers_of_H_real):
    v_c_s, v_p_s, t_c_s, t_p_s, f_c_s, f_p_s = [], [], [], [], [], []
    for i in range(numbers_of_H_real):
        H_real=H_reals[i]
        v_c,v_p,f_c,f_p=count_t_v_f_one(alpha_pri, alpha_comm, beta_pri, beta_comm,i, H_real, N_users)
        v_c_s.append(v_c)
        v_p_s.append(v_p)
        f_c_s.append(f_c)
        f_p_s.append(f_p)
    v_c_avg=count_avg_of_alpha_beta(v_c_s,numbers_of_H_real,N_users)
    v_p_avg=count_avg_of_alpha_beta(v_p_s,numbers_of_H_real,N_users)
    f_c_avg=count_avg_of_np(f_c_s,numbers_of_H_real,N_users)
    f_p_avg = count_avg_of_np(f_p_s, numbers_of_H_real, N_users)
    return v_c_avg,v_p_avg,f_c_avg,f_p_avg

def update_P_p(m_p,n_p,N_users,Nt,mu):
    P=[]
    part1=mu*np.eye(Nt)+1j*np.zeros((Nt,Nt))
    for i in range(N_users):
        part1 += m_p[i]
    for k in range(N_users):
        P_k=np.linalg.inv(part1)@n_p[k]
        P.append(P_k)
    return np.concatenate(P,axis=1)


def count_g_c(r_c,n_c,p_c,beta_mul_T_comm,N_users):
    g=[]
    for k in range(N_users):
       g_k=r_c[k]+np.real(2*np.conj(n_c[k].T)@p_c)[0][0]-np.real(beta_mul_T_comm[k])
       g.append(g_k)
    return g

def count_g_p(r_p,n_p,P_p,beta_mul_T_pri,N_users):
    g=[]
    for k in range(N_users):
       g_k=r_p[k] +np.real(2*np.conj(n_p[k].T)@P_p[:,k][:, np.newaxis])[0][0]-np.real(beta_mul_T_pri[k])
       g.append(g_k)
    return g


def count_T_avg(H_reals,numbers_of_H_real, P, N_users):
    T_pri_sum=[]
    for i in range(numbers_of_H_real):
        T_pri = count_T(H_reals[i], P, N_users)
        T_pri_sum.append(T_pri)
    T_pri_avg=count_avg_of_alpha_beta(T_pri_sum,numbers_of_H_real,N_users)
    return T_pri_avg

def quad_form(P,m,i,k):
    sum=((np.conj(P[:, i][:, np.newaxis].T)@m[k])@P[:, i][:, np.newaxis])[0][0]
    return sum

def count_beta_mul_T_avg(m_p, P, N_users):
    T_pri_sum = []
    for k in range(N_users):
        T_private_k = 0
        for i in range(N_users):
            T_private_k += quad_form(P,m_p,i,k)
        T_pri_sum.append(T_private_k)
    return T_pri_sum

def P_init(P_p,H,K,alpha):
    P=[]
    for i in range(K):
        P.append(H[:,i][:,np.newaxis ]/np.linalg.norm(H[:,i])*np.sqrt(P_p*alpha[i]))
    return np.concatenate(P,axis=1)
def P_total_init(Pt,H,K,alpha):
    P = P_init(Pt, H,K,alpha)
    return P
def order(H_estimate,H_reals,Nt,N_users,rho,tolerance,Pt,alpha,tolerance_inner,maxcount):
    P=P_total_init(Pt,H_estimate,N_users,alpha)
    flag=1
    obj_past=0
    count=0
    mu=10/Pt
    T1=time.time()
    numbers_of_H_real=len(H_reals)
    while(flag):
        count_inner = 0
        m_p, n_p, r_p=count_m_n_r_one(H_reals,numbers_of_H_real,P,N_users,Nt)
        mu=10/Pt
        while (count_inner <= maxcount):
            P=update_P_p(m_p,n_p,N_users,Nt,mu)
            mu_new = (np.trace(P@np.conj(P.T) ) ) / Pt  * mu
            e=count_e(mu,mu_new)
            mu=mu_new
            if np.linalg.norm(e)<tolerance_inner:
                break
            count_inner=count_inner+1
        beta_mul_T_pri = count_beta_mul_T_avg(m_p, P, N_users)
        g_pri=count_g_p(r_p,n_p,P,beta_mul_T_pri,N_users)
        obj = sum(g_pri)
        if abs(obj - obj_past) <= tolerance:
            flag = 0
        else:
            obj_past = obj
            count = count + 1
        if count >= 1000:
            break
    rates_pri=count_avg_rate(H_reals, numbers_of_H_real, P, N_users)
    obj = sum(rates_pri)
    T2=time.time()
    return obj,T2-T1

def count_avg_rate(H_reals, numbers_of_H_real, P, N_users):
    rates_comm_sum, rates_pri_sum = [], []
    for i in range(numbers_of_H_real):
        T_pri = count_T(H_reals[i], P, N_users)
        I_pri = count_I(T_pri, H_reals[i], P, N_users)
        rates_pri = count_rate(T_pri, I_pri, N_users)
        rates_pri_sum.append(rates_pri)

    rates_pri_avg = count_avg_of_alpha_beta(rates_pri_sum, numbers_of_H_real, N_users)
    return rates_pri_avg

def initial_H_random(seed, Nr, Nt,bias):
    H=[]
    np.random.seed(seed)
    for i in range(Nr):
        h=np.random.randn(Nt)+1j * np.random.randn(Nt)
        h=np.sqrt(bias[i])*h*1/np.sqrt(2)
        H.append(np.reshape(h,(Nt,1)))
    H=np.concatenate(H,axis=1)
    return H

if __name__ == "__main__":
    print('系统:', platform.system())
    T1 = time.time()
    Nt=4
    N_users=4
    rho=2
    tolerance=1e-3
    bias = [1, 1, 1,1]
    transmit_SNr=20
    Pt = 10 ** (transmit_SNr / 10)
    alpha=[0.25,0.25,0.25,0.4]
    maxcount=1000
    tolerance_inner=1e-5
    alpha_i=0.6
    P_e = Pt ** (-alpha_i)
    #P_e=0
    H_estimate=initial_H_random(1, N_users, Nt, bias)*np.sqrt(1-P_e)
    H_errors=[]
    H_reals=[]
    for seed_i in range(1000):
        H_error=initial_H_random(seed_i, N_users, Nt, bias)*np.sqrt(P_e)
        H_real=H_estimate+H_error
        H_errors.append(H_error)
        H_reals.append(H_real)
    SR=order(H_estimate,H_reals,Nt, N_users, rho, tolerance, Pt, alpha, tolerance_inner,maxcount)
    print(SR)
    T2 =time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
