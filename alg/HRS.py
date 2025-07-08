import numpy as np
import copy
import platform
def abs_square_of_Hk_mul_Pi(H,P,k,i):
    HP=abs(np.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])[0, 0])**2
    return HP
def square_of_Hk(H,k):
    H_square=np.matmul(H[:, k][:,np.newaxis], H[:, k].conjugate().T[ np.newaxis, :])
    return H_square
def Hk_mul_Pi(H,P,k,i):
    return np.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])[0,0]
def count_T(H,P,P_g,P_t,N_users,groups_dict_not_in,groups_dict_in):
    T_p=[]
    T_g=[]
    T_t=[]
    for k in range(N_users):
        T_private_k=1
        for i in range(N_users):
            T_private_k += abs_square_of_Hk_mul_Pi(H, P, k, i)
        group_not_in=groups_dict_not_in[k]
        for i in group_not_in:
            T_private_k+=abs_square_of_Hk_mul_Pi(H, P_g, k, i)
        T_p.append(T_private_k)
        T_private_k += abs_square_of_Hk_mul_Pi(H, P_g, k, groups_dict_in[k])
        T_g.append(T_private_k)
        T_private_k+=abs_square_of_Hk_mul_Pi(H,P_t,k,0)
        T_t.append(T_private_k)
    return T_p,T_g,T_t

def count_I(T_p,T_g,H,P,N_users):
    I_p,I_g,I_t=[],[],[]
    I_t = T_g
    I_g=T_p
    for k in range(N_users):
        I_private_k=T_p[k]-abs_square_of_Hk_mul_Pi(H,P,k,k)
        I_p.append(I_private_k)
    return I_p,I_g,I_t

def count_SINR(H,I_p,I_g,I_t,N_users,P,P_g,P_t,groups_dict_in):
    SINR_t,SINR_g,SINR_p=[],[],[]
    for k in range(N_users):
        SINR_g.append(abs_square_of_Hk_mul_Pi(H,P_g,k,groups_dict_in[k])/I_g[k])
        SINR_p.append(abs_square_of_Hk_mul_Pi(H,P,k,k)/I_p[k])
        SINR_t.append(abs_square_of_Hk_mul_Pi(H,P_t,k,0)/I_t[k])
    return SINR_t,SINR_g,SINR_p

def count_beta(H,alpha_p,alpha_g,alpha_t,N_users,P_g,P,P_t,T_t,T_p,T_g,groups_dict_in):
    beta_t,beta_g,beta_p=[],[],[]
    for k in range(N_users):
        beta_t_k=np.sqrt(1+alpha_t[k])*Hk_mul_Pi(H,P_t,k,0)/T_t[k]
        beta_p_k=np.sqrt(1+alpha_p[k])*Hk_mul_Pi(H,P,k,k)/T_p[k]
        beta_g_k=np.sqrt(1+alpha_g[k])*Hk_mul_Pi(H,P_g,k,groups_dict_in[k])/T_g[k]
        beta_t.append(beta_t_k)
        beta_p.append(beta_p_k)
        beta_g.append(beta_g_k)
    return beta_p,beta_g,beta_t

def count_y(g_comm):
    return min(g_comm)

def count_y_g(g_g,groups):
    y=[]
    index=[]
    groups_number=0
    for i in range(len(groups)):
        y.append(np.min(g_g[i]))
        index.append(np.argmin(g_g[i])+groups_number)
        groups_number+=groups[i]
    return y,index

def quad_form(P,m,i,k):
    sum=((np.conj(P[:, i][:, np.newaxis].T)@m[k])@P[:, i][:, np.newaxis])[0][0]
    return sum

def count_beta_mul_T_avg(m_p,m_t,m_g, P,P_g,P_t, N_users,groups_dict_not_in,groups_dict_in):
    T_t_sum, T_g_sum,T_p_sum = [], [], []
    for k in range(N_users):
        T_p=0
        T_g = 0
        T_t=0
        for i in range(N_users):
            T_p += quad_form(P,m_p,i,k)
            T_t += quad_form(P, m_t, i, k)
            T_g+=quad_form(P,m_g,i,k)
        group_not_in=groups_dict_not_in[k]
        for i in group_not_in:
            T_p+=quad_form(P_g,m_p,i,k)
            T_t += quad_form(P_g, m_t, i, k)
            T_g+=quad_form(P_g,m_g,i,k)
        T_p_sum.append(T_p)
        T_g += quad_form(P_g,m_g,groups_dict_in[k],k)
        T_t+=quad_form(P_g,m_t,groups_dict_in[k],k)
        T_g_sum.append(T_g)
        T_t+=quad_form(P_t,m_t,0,k)
        T_t_sum.append(T_t)
    return T_t_sum, T_g_sum,T_p_sum



def count_g_t(r_t,n_t,P_t,beta_mul_T_t,N_users):
    g=[]
    for k in range(N_users):
       g_k=r_t[k]+np.real(2*np.conj(n_t[k].T)@P_t)[0][0]-np.real(beta_mul_T_t[k])
       g.append(g_k)
    return g

def count_g_p(r_p,n_p,P,beta_mul_T_p,N_users):
    g=[]
    for k in range(N_users):
       g_k=r_p[k] +np.real(2*np.conj(n_p[k].T)@P[:,k][:, np.newaxis])[0][0]-np.real(beta_mul_T_p[k])
       g.append(g_k)
    return g
def count_g_g(r_g,n_g,P_g,beta_mul_T_g,G_groups,group_members):
    g=[]
    for i in range(G_groups):
        g_g=[]
        for k in group_members[i]:
            g_k=r_g[k] +np.real(2*np.conj(n_g[k].T)@P_g[:,i][:, np.newaxis])[0][0]-np.real(beta_mul_T_g[k])
            g_g.append(g_k)
        g.append(g_g)
    return g


def update_lambda_t(g_t,rou,lambda_t,N_users):
    lambda_new=[0 for i in range(len(lambda_t))]
    y=count_y(g_t)
    index = np.argmin(g_t)
    summ=0
    if y>=0:
        for k in range(N_users):
            lambda_new[k]=((y+rou)/(g_t[k]+rou))*lambda_t[k]
            summ+=lambda_t[k]-lambda_new[k]
    else:
        for k in range(N_users):
            lambda_new[k]=(rou-y)/(g_t[k]-2*y+rou)*lambda_t[k]
            summ+=lambda_t[k]-lambda_new[k]
    lambda_new[index]=lambda_t[index]+summ
    return lambda_new,summ

def update_lambda_g(g_g,rou,lambda_g,N_users,groups,group_members):
    lambda_new=[[0 for j in range(N_users)] for i in range(len(groups))]
    y_g,index_g=count_y_g(g_g,groups)
    summ=0
    group_numbers=0
    for i in range(len(y_g)):
        summ_i=0
        if y_g[i]>=0:
            for k in group_members[i]:
                lambda_new[i][k]=((y_g[i]+rou)/(g_g[i][k-group_numbers]+rou))*lambda_g[i][k]
                summ_i+=lambda_g[i][k]-lambda_new[i][k]
        else:
            for k in group_members[i]:
                lambda_new[i][k]=(rou-y_g[i])/(g_g[i][k-group_numbers]-2*y_g[i]+rou)*lambda_g[i][k]
                summ_i+=lambda_g[i][k]-lambda_new[i][k]
        group_numbers+=groups[i]
        lambda_new[i][index_g[i]]=lambda_g[i][index_g[i]]+summ_i
        summ+=abs(summ_i)
    return lambda_new,summ

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
def lambda_init(H,N_users,group_memebers):
    lambda_1=found_order(H,N_users)
    lambda_1 = [abs(l_i) ** 2 for l_i in lambda_1]
    sum_lambda = sum(lambda_1)
    lambda_1 = [l_i / sum_lambda for l_i in lambda_1]
    lambda_output_g = [[0 for j in range(N_users)] for i in range(len(group_memebers))]
    for i in range(len(group_memebers)):
        lambda_i=found_order(H[:,group_memebers[i]],len(group_memebers[i]))
        lambda_i=[abs(l_i)**2 for l_i in lambda_i]
        sum_lambda=sum(lambda_i)
        lambda_i=[l_i/sum_lambda for l_i in lambda_i]
        j=0
        for k in group_memebers[i]:
            lambda_output_g[i][k]=lambda_i[j]
            j+=1
    return lambda_output_g,lambda_1


def init_P(H,Pt,groups,groups_innerweight,alpha,N_users):
    P,P_g,P_t=[],[],[]
    power = Pt * alpha[-1]
    a,b,c=np.linalg.svd(H)
    P_t.append(np.sqrt(power)*a[:,0][:,np.newaxis ])
    power = Pt * alpha[1] / len(groups)
    for group_order in range(len(groups)):
        a, b, c = np.linalg.svd(H[:,groups[group_order]])
        P_g.append(np.sqrt(power) * a[:, 0][:, np.newaxis])
    power_pri=Pt*alpha[0]
    if N_users==4:
        for group_order in range(len(groups)):
            for i in range(len(groups[group_order])):
                power_pri_1 = power_pri * groups_innerweight[i]
                power_pri_1 = np.sqrt(power_pri_1)
                P.append(H[:, groups[group_order][i]][:, np.newaxis] / np.linalg.norm(H[:, groups[group_order][i]]) * power_pri_1)
    else:
        power_pri_1 = power_pri / N_users
        for i in range(N_users):
            P.append(H[:, i][:, np.newaxis] / np.linalg.norm(H[:, i]) * np.sqrt(power_pri_1))
    return np.concatenate(P,axis=1),np.concatenate(P_g,axis=1),np.concatenate(P_t,axis=1)

def count_e(mu,mu_new,summ_g,summ_t):
    e=0
    e+=abs(mu-mu_new)
    e+=abs(summ_g)
    e+=abs(summ_t)
    return e

def count_rate(T,I,N_users):
    rate=[]
    for k in range(N_users):
        rate.append(np.log2(T[k]/I[k]))
    return rate


def produce_group_members(group,N_users):
    group_members=[]
    group_notmembers=[]
    group_n=0
    for group_number in group:
        L=[]
        L2=[i for i in range(N_users)]
        for i in range(group_number):
            L.append(i+group_n)
            L2.remove(i+group_n)
        group_members.append(L)
        group_notmembers.append(L2)
        group_n+=group_number
    return group_members,group_notmembers

def produce_group_dict(group_members,groups,N_users):
    d=dict()
    d_not_in=dict()
    for i in range(N_users):
        l = []
        for group_i in range(len(groups)):
            if i in group_members[group_i]:
                d[i]=group_i
            else:
                l.append(group_i)
        d_not_in[i]=set(l)
    return d,d_not_in
import time
def count_alpha(T_t,T_p, T_g,I_p, I_g, I_t,N_users):
    alpha_p=[T_p[k]/I_p[k]-1 for k in range(N_users)]
    alpha_t=[T_t[k]/I_t[k]-1 for k in range(N_users)]
    alpha_g=[[T_g[k][i]/T_g[k][i]-1 for i in range(len(I_g[k]))] for k in range(N_users)]

    return alpha_p,alpha_t,alpha_g

def count_avg_alpha_beta_one(H,P, P_g, P_t,N_users,group_not_in_dict, group_in_dict):
    T_p, T_g, T_t = count_T(H, P, P_g, P_t, N_users, group_not_in_dict, group_in_dict)
    I_p, I_g, I_t = count_I(T_p, T_g, H, P, N_users)
    alpha_t, alpha_g, alpha_p = count_SINR(H, I_p, I_g, I_t, N_users, P, P_g, P_t, group_in_dict)
    beta_p, beta_g, beta_t = count_beta(H, alpha_p, alpha_g, alpha_t, N_users, P_g, P, P_t, T_t, T_p, T_g,
                                        group_in_dict)
    return alpha_t, alpha_g, alpha_p,beta_p, beta_g, beta_t

def count_m_n_r_one(H_reals,numbers_of_H_real,P, P_g, P_t,N_users,Nt,group_not_in_dict, group_in_dict):
    m_p_s,m_t_s,m_g_s=[np.zeros((Nt,Nt)) for k in range(N_users)],[np.zeros((Nt,Nt)) for k in range(N_users)],[np.zeros((Nt,Nt)) for k in range(N_users)]
    n_p_s,n_t_s,n_g_s=[np.zeros((Nt,1)) for k in range(N_users)],[np.zeros((Nt,1)) for k in range(N_users)],[np.zeros((Nt,1)) for k in range(N_users)]
    r_t_s,r_p_s,r_g_s=[0 for k in range(N_users)],[0 for k in range(N_users)],[0 for k in range(N_users)]
    for i in range(numbers_of_H_real):
        H_real=H_reals[i]
        alpha_t, alpha_g, alpha_p,beta_p, beta_g, beta_t=count_avg_alpha_beta_one(H_real,P, P_g, P_t,N_users,group_not_in_dict, group_in_dict)
        beta_p_square=[abs(beta_p[k])**2 for k in range(N_users)]
        beta_g_square=[abs(beta_g[k])**2 for k in range(N_users)]
        beta_t_square=[abs(beta_t[k])**2 for k in range(N_users)]
        square_of_H=[square_of_Hk(H_real,k) for k in range(N_users)]
        m_p=[beta_p_square[k]*square_of_H[k] for k in range(N_users)]
        n_p=[np.sqrt(1+alpha_p[k])*beta_p[k]*H_real[:, k][:, np.newaxis] for k in range(N_users)]
        m_t=[beta_t_square[k]*square_of_H[k] for k in range(N_users)]
        n_t=[np.sqrt(1+alpha_t[k])*beta_t[k]*H_real[:, k][:, np.newaxis] for k in range(N_users)]
        m_g=[beta_g_square[k]*square_of_H[k] for k in range(N_users)]
        n_g = [np.sqrt(1 + alpha_g[k]) * beta_g[k] * H_real[:, k][:, np.newaxis] for k in range(N_users)]
        r_t=[np.log2(1+alpha_t[k])-beta_t_square[k]-alpha_t[k] for k in range(N_users)]
        r_p=[np.log2(1+alpha_p[k])-beta_p_square[k]-alpha_p[k] for k in range(N_users)]
        r_g = [np.log2(1 + alpha_g[k]) - beta_g_square[k] - alpha_g[k] for k in range(N_users)]
        m_p_s=[m_p_s[k]+m_p[k] for k in range(N_users)]
        m_t_s=[m_t_s[k]+m_t[k] for k in range(N_users)]
        m_g_s=[m_g_s[k]+m_g[k] for k in range(N_users)]
        n_p_s=[n_p_s[k]+n_p[k] for k in range(N_users)]
        n_t_s=[n_t_s[k]+n_t[k] for k in range(N_users)]
        n_g_s=[n_g_s[k]+n_g[k] for k in range(N_users)]
        r_t_s=[r_t_s[k]+r_t[k] for k in range(N_users)]
        r_p_s=[r_p_s[k]+r_p[k] for k in range(N_users)]
        r_g_s=[r_g_s[k]+r_g[k] for k in range(N_users)]
    m_p_avg = [m_p_s[k]/numbers_of_H_real for k in range(N_users)]
    m_t_avg = [m_t_s[k]/numbers_of_H_real for k in range(N_users)]
    m_g_avg = [m_g_s[k]/numbers_of_H_real for k in range(N_users)]
    n_p_avg= [n_p_s[k]/numbers_of_H_real for k in range(N_users)]
    n_t_avg = [n_t_s[k]/numbers_of_H_real for k in range(N_users)]
    n_g_avg = [n_g_s[k]/numbers_of_H_real for k in range(N_users)]
    r_t_avg= [r_t_s[k]/numbers_of_H_real for k in range(N_users)]
    r_p_avg = [r_p_s[k] /numbers_of_H_real for k in range(N_users)]
    r_g_avg = [r_g_s[k] /numbers_of_H_real for k in range(N_users)]
    return m_p_avg,m_t_avg,m_g_avg,n_p_avg,n_t_avg,n_g_avg,r_t_avg,r_p_avg,r_g_avg

def update_Pp(lambda_g,lambda_t,m_p,n_p,m_g,m_t,N_users,Nt,mu,G_groups,group_members):
    P=[]
    part1=mu*np.eye(Nt)+1j*np.zeros((Nt,Nt))
    for i in range(N_users):
        part1 += lambda_t[i]*m_t[i]+ m_p[i]
    for g in range(G_groups):
        for i in group_members[g]:
            part1 += m_g[i] * lambda_g[g][i]
    part1_inv=np.linalg.inv(part1)
    for k in range(N_users):
        P_k=part1_inv@n_p[k]
        P.append(P_k)
    return np.concatenate(P,axis=1)

def update_Pt(lambda_t,m_t,n_t,N_users,Nt,mu):
    part1=mu*np.eye(Nt)+1j*np.zeros((Nt,Nt))
    part2=np.zeros_like(n_t[0])
    for i in range(N_users):
        part1+=lambda_t[i]*m_t[i]
        part2+=lambda_t[i]*n_t[i]
    P_t=np.linalg.inv(part1)@part2
    return P_t

def update_Pg(lambda_g,lambda_t,n_g,m_t,m_g,m_p,N_users,Nt,mu,group_not_members,group_members,G_groups):
    P=[]
    part1_base=mu*np.eye(Nt)+1j*np.zeros((Nt,Nt))
    for i in range(N_users):
        part1_base += lambda_t[i]*m_t[i]
    for g in range(G_groups):
        for i in group_members[g]:
            part1_base+=m_g[i]*lambda_g[g][i]

    for g in range(G_groups):
        part1=copy.deepcopy(part1_base)
        for i in group_not_members[g]:
            part1+=m_p[i]
        part2=np.zeros_like(n_g[0])
        for i in group_members[g]:
            part2+=n_g[i]*lambda_g[g][i]
        P_g=np.linalg.inv(part1)@part2
        P.append(P_g)
    return np.concatenate(P,axis=1)

def order(H_estimate,H_reals,group,Nt, N_users, rho, tolerance, Pt, alpha,alpha_inner, tolerance_inner,maxcount):
    Z=[]
    G_groups=len(group)
    group_members,group_not_members=produce_group_members(group,N_users)
    group_in_dict,group_not_in_dict=produce_group_dict(group_members,group,N_users)
    lambda_g,lambda_t=lambda_init(H_estimate,N_users,group_members)
    mu=10/Pt
    P,P_g,P_t=init_P(H_estimate, Pt, group_members, alpha_inner, alpha,N_users)
    flag=1
    obj_past=0
    count=0
    numbers_of_H_real=len(H_reals)
    T1=time.time()
    while(flag):
        m_p,m_t,m_g,n_p,n_t,n_g,r_t,r_p,r_g=\
        count_m_n_r_one(H_reals, numbers_of_H_real, P, P_g, P_t, N_users, Nt, group_not_in_dict, group_in_dict)
        count_inner = 0
        error=[]
        while (count_inner <= maxcount):
            P_t=update_Pt(lambda_t,m_t,n_t,N_users,Nt,mu)
            P_g=update_Pg(lambda_g,lambda_t,n_g,m_t,m_g,m_p,N_users,Nt,mu,group_not_members,group_members,G_groups)
            P=update_Pp(lambda_g,lambda_t,m_p,n_p,m_g,m_t,N_users,Nt,mu,G_groups,group_members)
            beta_mul_T_t, beta_mul_T_g,beta_mul_T_p=count_beta_mul_T_avg(m_p, m_t, m_g, P, P_g, P_t, N_users, group_not_in_dict, group_in_dict)
            g_t=count_g_t(r_t,n_t,P_t,beta_mul_T_t,N_users)
            g_g=count_g_g(r_g,n_g,P_g,beta_mul_T_g,G_groups,group_members)
            lambda_new_t,summ_t=update_lambda_t(g_t,rho,lambda_t,N_users)
            lambda_new_g,summ_g=update_lambda_g(g_g,rho,lambda_g,N_users,group,group_members)
            mu_new = (np.trace(P@np.conj(P.T)+P_t@np.conj(P_t.T)+P_g@np.conj(P_g.T)) ) / Pt  * mu
            e=count_e(mu,mu_new,summ_g,summ_t)
            error.append(e)
            lambda_g=lambda_new_g
            lambda_t=lambda_new_t
            mu=mu_new
            if np.linalg.norm(e)<tolerance_inner:
                break
            count_inner=count_inner+1
        beta_mul_T_t, beta_mul_T_g, beta_mul_T_p = count_beta_mul_T_avg(m_p, m_t, m_g, P, P_g, P_t, N_users,
                                                                        group_not_in_dict, group_in_dict)
        g_t = count_g_t(r_t, n_t, P_t, beta_mul_T_t, N_users)
        g_g = count_g_g(r_g,n_g,P_g,beta_mul_T_g,G_groups,group_members)
        g_p=count_g_p(r_p, n_p, P, beta_mul_T_p, N_users)
        obj = sum(g_p) + min(g_t)
        for i in range(G_groups):
            obj += min(g_g[i])
        if abs(obj - obj_past) <= tolerance:
            flag = 0
        else:
            obj_past = obj
            count = count + 1
        if count >= 1000:
            break
    rates_p,rates_g,rates_t=count_avg_rate(H_reals, numbers_of_H_real, P, P_g, P_t, group_not_in_dict, group_in_dict, N_users)
    obj = sum(rates_p) + min(rates_t)
    for i in range(G_groups):
        L=[]
        for k in group_members[i]:
            L.append(rates_g[k])
        obj += min(L)
    T2 = time.time()
    return obj,T2-T1

def produce_H(seed,N_users,Nt, bias,P_e,N_random_H):
    H_estimate = initial_H_random(seed, N_users, Nt, bias) * np.sqrt(1 - P_e)
    H_errors = []
    H_reals = []
    for seed_i in range(N_random_H):
        H_error = initial_H_random(seed_i, N_users, Nt, bias) * np.sqrt(P_e)
        H_real = H_estimate + H_error
        H_errors.append(H_error)
        H_reals.append(H_real)
    return H_estimate,H_reals,H_errors
def initial_H_random(seed, Nr, Nt,bias):
    H=[]
    np.random.seed(seed)
    for i in range(Nr):
        h=np.random.randn(Nt)+1j * np.random.randn(Nt)
        h=np.sqrt(bias[i])*h*1/np.sqrt(2)
        H.append(np.reshape(h,(Nt,1)))
    H=np.concatenate(H,axis=1)
    return H

def count_avg_rate(H_reals, numbers_of_H_real, P, P_g, P_t,group_not_in_dict, group_in_dict, N_users):
    rates_t_sum, rates_g_sum ,rates_p_sum = [0 for i in range(N_users)], [0 for i in range(N_users)], [0 for i in range(N_users)]
    for i in range(numbers_of_H_real):
        H_real=H_reals[i]
        T_p, T_g, T_t = count_T(H_real, P, P_g, P_t, N_users, group_not_in_dict, group_in_dict)
        I_p, I_g, I_t = count_I(T_p, T_g, H_real, P, N_users)
        rates_t = count_rate(T_t, I_t, N_users)
        rates_g = count_rate(T_g, I_g, N_users)
        rates_p = count_rate(T_p, I_p, N_users)
        for k in range(N_users):
            rates_t_sum[k]+=rates_t[k]
            rates_g_sum[k]+=rates_g[k]
            rates_p_sum[k]+=rates_p[k]
    rates_p_avg = [rates_p_sum[k]/numbers_of_H_real for k in range(N_users)]
    rates_g_avg = [rates_g_sum[k]/numbers_of_H_real for k in range(N_users)]
    rates_t_avg = [rates_t_sum[k]/numbers_of_H_real for k in range(N_users)]

    return rates_p_avg,rates_g_avg,rates_t_avg

if __name__ == "__main__":
    print('系统:', platform.system())
    T1 = time.time()
    group=[2,2]
    alpha= [0.1,0.1,0.8]
    alpha_inner=[0.4,0.1]
    N_users=4
    Nt=4
    rho=2
    tolerance=1e-3
    transmit_SNr=20
    Pt = 10 ** (transmit_SNr / 10)
    tolerance_inner=10**(-5)
    maxcount=1000
    bias=[1,1,1,1]
    alpha_i = 0.6
    P_e = Pt ** (-alpha_i)
    N_random_H=1000
    H_estimate,H_reals,H_errors=produce_H(1, N_users, Nt, bias, P_e, N_random_H)
    SR=order( H_estimate,H_reals,group, Nt, N_users, rho, tolerance, Pt, alpha, alpha_inner, tolerance_inner, maxcount)
    print(SR)
    T2 =time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
