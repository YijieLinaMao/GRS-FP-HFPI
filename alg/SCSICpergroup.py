import numpy as np
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

def count_T(H,P,groups_members,N_users):
    group_number=0
    T=[]
    for group in groups_members:
        group_length=len(group)
        for k in group:
            T_k = []
            for i in range(group_number,k+1):
                T_k_i = 1
                for j in range(i, group_number+group_length):
                    T_k_i += abs_square_of_Hk_mul_Pi(H, P, k, j)
                for j in range(group_number):
                    T_k_i += abs_square_of_Hk_mul_Pi(H, P, k, j)
                for j in range(group_number+group_length,N_users):
                    T_k_i += abs_square_of_Hk_mul_Pi(H, P, k, j)
                T_k.append(T_k_i)
            T.append(T_k)
        group_number+=group_length
    return T

def count_I(T,H,P,N_users):
    I=[]
    for k in range(N_users):
        I_k=T[k][1:len(T[k])]
        I_k.append(T[k][-1]-abs_square_of_Hk_mul_Pi(H,P,k,k))
        I.append(I_k)
    return I

def reshape_T_and_I(T_I,groups_members):
    group_number=0
    for group in groups_members:
        group_length=len(group)
        for k in group:
            T_I[k]=[0 for i in range(group_number)]+T_I[k]
        group_number+=group_length
    return T_I

def count_SINR(I,P,H,groups_members):
    SINR=[]
    group_number=0
    for group in groups_members:
        group_length=len(group)
        for k in group:
            SINR_k = []
            for i in range(group_number,k+1):
                SINR_i_k = abs_square_of_Hk_mul_Pi(H, P, k, i) / I[k][i]
                SINR_k.append(SINR_i_k)
            SINR.append(SINR_k)
        group_number += group_length
    return SINR

def count_beta(alpha,P,H,T,groups_members):
    beta=[]
    group_number=0
    for group in groups_members:
        group_length=len(group)
        for k in group:
            beta_k = []
            for i in range(group_number,k+1):
                beta_ki=np.sqrt(1+alpha[k][i])*Hk_mul_Pi(H,P,k,i)/T[k][i]
                beta_k.append(beta_ki)
            beta.append(beta_k)
        group_number += group_length
    return beta

def count_first_part_of_update_P(j,lambda_i,m,mu,Nt,all_members_in_my_group,all_members_not_in_my_group):
    summ=0
    group_in=all_members_in_my_group[j]
    group_not_in=all_members_not_in_my_group[j]
    for i in group_in:
        if i<j+1:
            for k in group_in:
                if k>=i:
                    summ+=lambda_i[i][k]*m[k][i]
    for i in group_not_in:
        for k in group_not_in:
            if k >= i:
                summ += lambda_i[i][k] *m[k][i]
    return summ+mu*np.eye(Nt)

def count_second_part_of_update_P(j,n,lambda_i,all_members_in_my_group):
    summ=0
    group_in=all_members_in_my_group[j]
    for k in group_in:
        if k >=j:
            summ+=n[k][j]*lambda_i[j][k]
    return summ

def update_P(m,n,lambda_i,mu,N_users,Nt,all_members_in_my_group,all_members_not_in_my_group):
    P=[]
    for j in range(N_users):
        part_1=count_first_part_of_update_P(j,lambda_i,m,mu,Nt,all_members_in_my_group,all_members_not_in_my_group)
        part_2=count_second_part_of_update_P(j,n,lambda_i,all_members_in_my_group)
        P.append(np.linalg.inv(part_1)@part_2)
    return np.concatenate(P,axis=1)


def count_y(g,groups_members):
    index=[]
    minn=[]
    group_number=0
    for group in groups_members:
        group_length=len(group)
        for j in group:
            g_j = [g[k][j-group_number] for k in range(j, group_number+group_length)]
            minn.append(min(g_j))
            index.append(np.argmin(g_j) + j)
        group_number+=group_length
    return minn,index


def count_g(P,n,r,T,groups_members):
    g=[]
    group_number=0
    for group in groups_members:
        group_length=len(group)
        for k in group:
            g_k = []
            for i in range(group_number,k+1):
                g_part_1 = r[k][i]
                g_part_2=np.real(2 * np.conj(n[k][i].T) @ P[:, i][:, np.newaxis])[0][0]
                g_part_3 =np.real(T[k][i])
                g_k.append(g_part_1 + g_part_2 - g_part_3)
            g.append(g_k)
        group_number+=group_length
    return g

def update_lambda(g,rou,lambda_i,group_members):
    lambda_new=[[0 for j in range(len(lambda_i[i]))] for i in range(len(lambda_i))]
    y,index=count_y(g,group_members)
    summ_list=[]
    group_numbers=0
    for group in group_members:
        group_length=len(group)
        for j in group:
            summ = 0
            for k in range(j,group_numbers+group_length):
                if y[j] >= 0:
                    lambda_new[j][k] = ((y[j] + rou) / (g[k][j-group_numbers] + rou)) * lambda_i[j][k]
                    summ += lambda_i[j][k] - lambda_new[j][k]
                else:
                    lambda_new[j][k] = (rou - y[j]) / (g[k][j-group_numbers] - 2 * y[j] + rou) * lambda_i[j][k]
                    summ += lambda_i[j][k] - lambda_new[j][k]
            lambda_new[j][index[j]] = lambda_i[j][index[j]] + summ
            summ_list.append(summ)
        group_numbers+=group_length

    return lambda_new,summ_list

def init_P(H,Pt,alpha,group):
    P=[]
    before_group_number=0
    for group_number in group:
        nums = [i for i in range(before_group_number, group_number+before_group_number)]
        i=0
        while(len(nums)>1):
            power = Pt * alpha[i+before_group_number]
            a, b, c = np.linalg.svd(H[:, nums])
            nums.pop(0)
            P.append(np.sqrt(power)*a[:,0][:,np.newaxis ])
            i+=1
        power = Pt * alpha[i+before_group_number]
        P.append(H[:,i+before_group_number][:,np.newaxis ]/np.linalg.norm(H[:,i+before_group_number])*np.sqrt(power))
        before_group_number+=group_number
    return np.concatenate(P,axis=1)

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
import copy
def lambda_init(H,N_users,group_members):
    lambda_output=[]
    L=[i for i in range(N_users)]
    group_numbers=0
    for group in group_members:
        t=copy.deepcopy(group)
        j=0
        for i in group:
            lambda_i = found_order(H[:,t], len(t))
            t.remove(i)
            lambda_i=[abs(l_i)**2 for l_i in lambda_i]
            sum_lambda=sum(lambda_i)
            lambda_i=[l_i/sum_lambda for l_i in lambda_i]
            zeros = [0 for i in range(group_numbers+j)]
            lambda_output.append(zeros + lambda_i)
            j+=1
        group_numbers+=len(group)
    return lambda_output



def count_e(mu,mu_new,summ_list):
    e=0
    e += abs(mu-mu_new)
    for summ in summ_list:
        e+=abs(summ)
    return e

def count_pri_rate(T,I,N_users):
    rate=[]
    for k in range(N_users):
        rate.append(np.log2(T[k]/I[k]))
    return rate

def count_rate(T,I,N_users):
    rate=[]
    for k in range(N_users):
        rate_k = []
        for i in range(len(I[k])):
            rate_k.append(np.log2(T[k][i]/I[k][i]))
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

def find_members(groups,N_users):
    group_members=[]
    all_members_in_my_group=dict()
    all_members_not_in_my_group=dict()
    group_number=0
    for group_i in groups:
        group_members_i=[]
        group_members_else=[i for i in range(N_users)]
        for i in range(group_i):
            group_members_i.append(group_number+i)
            group_members_else.remove(group_number+i)
        t=tuple(group_members_i)
        t2=tuple(group_members_else)
        for i in range(group_i):
            all_members_in_my_group[group_number+i]=t
            all_members_not_in_my_group[group_number+i]=t2
        group_number=group_number+group_i
        group_members.append(group_members_i)
    return group_members,all_members_in_my_group,all_members_not_in_my_group
def count_avg_rate(H_reals, numbers_of_H_real, P, N_users, group_members):
    T = count_T(H_reals[0], P, group_members, N_users)
    I = count_I(T, H_reals[0], P, N_users)
    rates_sum = count_rate(T, I, N_users)
    for i in range(1,numbers_of_H_real):
        H_real=H_reals[i]
        T = count_T(H_real, P, group_members, N_users)
        I = count_I(T, H_real, P, N_users)
        rates = count_rate(T, I, N_users)
        for k in range(N_users):
            for i in range(len(rates_sum[k])):
                rates_sum[k][i]+=rates[k][i]
    rates_avg = [[rates_sum[k][i]/numbers_of_H_real for i in range(len(rates_sum[k]))] for k in range(N_users)]
    return rates_avg

def count_avg_alpha_beta_one(H,P,N_users, group_members):
    T = count_T(H, P, group_members, N_users)
    I = count_I(T, H, P, N_users)
    T = reshape_T_and_I(T, group_members)
    I = reshape_T_and_I(I, group_members)
    alpha = count_SINR(I, P, H, group_members)
    alpha = reshape_T_and_I(alpha, group_members)
    beta = count_beta(alpha, P, H, T, group_members)
    beta = reshape_T_and_I(beta, group_members)

    return alpha,beta

def count_m_n_r(H_reals,numbers_of_H_real,P,N_users,Nt, group_members):
    m_s=[[np.zeros((Nt,Nt)) for i in range(k+1)]for k in range(N_users)]
    n_s=[[np.zeros((Nt,1)) for i in range(k+1)]for k in range(N_users)]
    r_s=[[0 for i in range(k+1)]for k in range(N_users)]
    for i in range(numbers_of_H_real):
        H_real=H_reals[i]
        alpha,beta=count_avg_alpha_beta_one(H_real,P,N_users, group_members)
        beta_square=[[abs(beta[k][i])**2 for i in range(k+1)]for k in range(N_users)]
        square_of_H=[square_of_Hk(H_real,k) for k in range(N_users)]
        m=[[beta_square[k][i]*square_of_H[k] for i in range(k+1)] for k in range(N_users)]
        n=[[np.sqrt(1+alpha[k][i])*beta[k][i]*H_real[:, k][:, np.newaxis] for i in range(k+1)] for k in range(N_users)]
        r=[[np.log2(1+alpha[k][i])-beta_square[k][i]-alpha[k][i] for i in range(k+1)] for k in range(N_users)]
        m_s=[[m_s[k][i]+m[k][i] for i in range(k+1)]for k in range(N_users)]
        n_s=[[n_s[k][i]+n[k][i]for i in range(k+1)] for k in range(N_users)]
        r_s=[[r_s[k][i]+r[k][i]for i in range(k+1)] for k in range(N_users)]
    m_avg = [[m_s[k][i]/numbers_of_H_real for i in range(k+1)] for k in range(N_users)]
    n_avg = [[n_s[k][i]/numbers_of_H_real for i in range(k+1)]  for k in range(N_users)]
    r_avg= [[r_s[k][i]/numbers_of_H_real for i in range(k+1)] for k in range(N_users)]
    return m_avg,n_avg,r_avg
def quad_form(P,m,i,k,j):
    sum=((np.conj(P[:, i][:, np.newaxis].T)@m[k][j])@P[:, i][:, np.newaxis])[0][0]
    return sum
def count_beta_mul_T_avg(m,P,N_users,groups_members):
    group_number = 0
    T = []
    for group in groups_members:
        group_length = len(group)
        for k in group:
            T_k = [0 for i in range(group_number, k + 1)]
            for i in range(group_number, k + 1):
                for j in range(i, group_number + group_length):
                    T_k[i-group_number] += quad_form(P,m,j,k,i)
                for j in range(group_number):
                    T_k[i-group_number] += quad_form(P,m,j,k,i)
                for j in range(group_number + group_length, N_users):
                    T_k[i-group_number] += quad_form(P,m,j,k,i)
            T.append(T_k)
        group_number += group_length
    return T

def order(H_estimate,H_reals, groups,Nt,N_users,rho,tolerance,Pt,alpha,tolerance_inner,maxcount):
    group_members,all_members_in_my_group,all_members_not_in_my_group=find_members(groups,N_users)
    flag=1
    obj_past=0
    count=0
    lambda_i=lambda_init(H_estimate,N_users,group_members)
    mu=10/Pt
    #Z=[]
    P=init_P(H_estimate,Pt,alpha,groups)
    T1=time.time()
    numbers_of_H_real=len(H_reals)
    while(flag):
        mu=10/Pt
        count_inner = 0
        m,n,r=count_m_n_r(H_reals, numbers_of_H_real, P, N_users, Nt, group_members)
        while (count_inner <= maxcount):
            P=update_P(m,n,lambda_i,mu,N_users,Nt,all_members_in_my_group,all_members_not_in_my_group)
            T=count_beta_mul_T_avg(m,P,N_users,group_members)
            T = reshape_T_and_I(T, group_members)
            g= count_g(P,n,r,T,group_members)
            lambda_new,summ=update_lambda(g,rho,lambda_i,group_members)
            mu_new = (np.trace(P@np.conj(P.T)) ) / Pt  * mu
            e=count_e(mu,mu_new,summ)
            lambda_i=lambda_new
            mu=mu_new
            if np.linalg.norm(e)<tolerance_inner:
                break
            count_inner=count_inner+1
        T = count_beta_mul_T_avg(m, P, N_users, group_members)
        T = reshape_T_and_I(T, group_members)
        g = count_g(P, n, r, T, group_members)
        minn,index=count_y(g, group_members)
        obj=sum(minn)
        if abs(obj - obj_past) <= tolerance:
            flag = 0
        else:
            obj_past = obj
            count = count + 1
        if count >= 10000:
            break
    rates = count_avg_rate(H_reals, numbers_of_H_real, P, N_users, group_members)
    rates_tot,index= count_y(rates,group_members)
    sum_rates=sum(rates_tot)
    T2=time.time()
    return sum_rates,T2-T1
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
    transmit_SNr=20
    Pt = 10 ** (transmit_SNr / 10)
    groups=[2,2]
    alpha=[0.3, 0.2, 0.3, 0.2]
    maxcount=1000
    tolerance_inner=10**(-5)
    alpha_i=0.6
    P_e = Pt ** (-alpha_i)
    bias=[1,1,1,1]
    H_estimate=initial_H_random(1, N_users, Nt, bias)*np.sqrt(1-P_e)
    H_errors=[]
    H_reals=[]
    for seed_i in range(1000):
        H_error=initial_H_random(seed_i, N_users, Nt, bias)*np.sqrt(P_e)
        H_real=H_estimate+H_error
        H_errors.append(H_error)
        H_reals.append(H_real)
    SR=order(H_estimate,H_reals, groups,Nt,N_users,rho,tolerance,Pt,alpha,tolerance_inner,maxcount)
    print(SR)
    T2 =time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
