import numpy as np
import copy
import platform
from itertools import combinations

def abs_square_of_Hk_mul_Pi(H,P,k,i):
    HP=abs(np.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])[0, 0])**2
    return HP

def square_of_Hk(H,k):
    H_square=np.matmul(H[:, k][:,np.newaxis], H[:, k].conjugate().T[ np.newaxis, :])
    return H_square

def Hk_mul_Pi(H,P,k,i):
    return np.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])[0,0]

def count_T_2(H,P,P_c,N_users,order_list,combs_number_all):
    T_p=[]
    T_c=[[0 for i in range(combs_number_all)] for j in range(N_users)]
    for k in range(N_users):
        T_private_k=1
        for i in range(N_users):
            T_private_k += abs_square_of_Hk_mul_Pi(H, P, k, i)
        for i in order_list[k][0]:
            T_private_k += abs_square_of_Hk_mul_Pi(H, P_c, k, i)
        T_p.append(T_private_k)
        for i in order_list[k][1]:
            T_private_k += abs_square_of_Hk_mul_Pi(H, P_c, k, i)
            T_c[k][i]=T_private_k
    return T_p,T_c

def count_T(H,P,P_c,N_users,order_list):
    T_p=[]
    T_c=[]
    for k in range(N_users):
        T_private_k=1
        T_c_k=[]
        for i in range(N_users):
            T_private_k += abs_square_of_Hk_mul_Pi(H, P, k, i)
        for i in order_list[k][0]:
            T_private_k += abs_square_of_Hk_mul_Pi(H, P_c, k, i)
        T_p.append(T_private_k)
        order_list_k_reverse=reversed(order_list[k][1])
        for i in order_list_k_reverse:
            T_private_k += abs_square_of_Hk_mul_Pi(H, P_c, k, i)
            T_c_k.append(T_private_k)
        T_c_k.reverse()
        T_c.append(T_c_k)
    return T_p,T_c

def count_I(T_p,T_c,H,P,N_users):
    I_p,I_c=[],[]
    for k in range(N_users):
        I_c_k=T_c[k][1:len(T_c[k])]
        I_c_k.append(T_p[k])
        I_c.append(I_c_k)
        I_private_k=T_p[k]-abs_square_of_Hk_mul_Pi(H,P,k,k)
        I_p.append(I_private_k)
    return I_p,I_c

def count_SINR(I_comm,I_pri,N_users,p_c,P,H,order_list):
    SINR_comm,SINR_pri=[],[]
    for k in range(N_users):
        SINR_comm_k = []
        for i in range(len(order_list[k][1])):
            SINR_comm_k.append(abs_square_of_Hk_mul_Pi(H,p_c,k,order_list[k][1][i])/I_comm[k][i])
        SINR_comm.append(SINR_comm_k)
        SINR_pri.append(abs_square_of_Hk_mul_Pi(H,P,k,k)/I_pri[k])
    return SINR_comm,SINR_pri

def count_alpha(SINR_comm,SINR_pri):
    return SINR_comm,SINR_pri

def count_beta(alpha_comm,alpha_pri,N_users,p_c,P,H,T_comm,T_pri,order_list):
    beta_comm,beta_pri=[],[]
    for k in range(N_users):
        beta_comm_k = []
        for i in range(len(order_list[k][1])):
            beta_comm_ki=np.sqrt(1+alpha_comm[k][i])*Hk_mul_Pi(H,p_c,k,order_list[k][1][i])/T_comm[k][i]
            beta_comm_k.append(beta_comm_ki)
        beta_pri_k=np.sqrt(1+alpha_pri[k])*Hk_mul_Pi(H,P,k,k)/T_pri[k]
        beta_comm.append(beta_comm_k)
        beta_pri.append(beta_pri_k)
    return beta_comm,beta_pri

def count_first_part_of_update_P_comm(i,lambda_i,beta_pri,beta_comm,H,mu,Nt,order_list,dict_users_not_in,dict_users_in):
    summ=0
    for j in dict_users_not_in[i]:
        summ+=(abs(beta_pri[j])**2)*square_of_Hk(H,j)
    for m in dict_users_in[i]:
        order_list_m=order_list[m][1]
        index = order_list_m.index(i)
        for j in range(index+1):
            summ+=abs(beta_comm[m][j])**2*square_of_Hk(H,m)*lambda_i[order_list[m][1][j]][m]
    for m in dict_users_not_in[i]:
        for j in range(len(order_list[m][1])):
            summ+=abs(beta_comm[m][j])**2*square_of_Hk(H,m)*lambda_i[order_list[m][1][j]][m]
    return summ+mu*np.eye(Nt)

def count_second_part_of_update_P_comm(i,alpha_comm,beta_comm,lambda_i,H,dict_users_in,order_list):
    summ=0
    for k in dict_users_in[i]:
        index = order_list[k][1].index(i)
        summ+=np.sqrt(1+alpha_comm[k][index])*beta_comm[k][index]*lambda_i[i][k]* H[:, k][:, np.newaxis]
    return summ

def update_P_comm(alpha_comm,beta_comm,beta_pri,lambda_i,mu,H,combs_number_all,Nt,order_list,dict_users_not_in,dict_users_in):
    P_c=[]
    for i in range(combs_number_all):
        part_1=count_first_part_of_update_P_comm(i,lambda_i,beta_pri,beta_comm,H,mu,Nt,order_list,dict_users_not_in,dict_users_in)
        part_2=count_second_part_of_update_P_comm(i,alpha_comm,beta_comm,lambda_i,H,dict_users_in,order_list)
        P_c.append(np.linalg.inv(part_1)@part_2)
    return np.concatenate(P_c,axis=1)

def count_first_part_of_update_P_pri(lambda_i,beta_pri,beta_comm,H,mu,N_users,Nt,order_list):
    summ=0
    for k in range(N_users):
        first_part=abs(beta_pri[k])**2
        second_part=0
        for i in range(len(order_list[k][1])):
            second_part+=(lambda_i[order_list[k][1][i]][k])*(abs(beta_comm[k][i])**2)
        summ+=(first_part+second_part)*square_of_Hk(H,k)
    return summ+mu*np.eye(Nt)

def count_second_part_of_update_P_pri(alpha_pri,beta_pri,H,k):
    summ=np.sqrt(1+alpha_pri[k])*beta_pri[k]*H[:, k][:, np.newaxis]
    return summ

def update_P_pri(alpha_pri,beta_comm,beta_pri,lambda_i,H,N_users,Nt,combs_number_all,mu):
    P=[]
    part_1 = count_first_part_of_update_P_pri(lambda_i, beta_pri, beta_comm, H, mu, N_users, Nt, combs_number_all)
    part_1_inv=np.linalg.inv(part_1)
    for k in range(N_users):
        part_2 = count_second_part_of_update_P_pri(alpha_pri,beta_pri,H,k)
        P.append(part_1_inv@part_2)
    return np.concatenate(P,axis=1)

def count_y(g_comm,combs_number_all,dict_users_in):
    index=[]
    y=[]
    for j in range(combs_number_all):
        y.append(np.min(g_comm[j]))
        i=np.argmin(g_comm[j])
        users=list(dict_users_in[j])
        index.append(users[i])
    return y,index

def count_g_pri(H,P,alpha_pri,beta_pri,N_users,T_pri):
    g=[]
    for k in range(N_users):
        g_part_1=np.log2(1+alpha_pri[k])-alpha_pri[k]
        g_part_2=2*np.sqrt(1+alpha_pri[k])*np.real(np.conj(beta_pri[k].T)*Hk_mul_Pi(H,P,k,k))
        g_part_3=(abs(beta_pri[k])**2)*T_pri[k]
        g.append(g_part_1+g_part_2-g_part_3)
    return g

def count_g_comm(H,p_c,alpha_comm,beta_comm,T_comm,dict_users_in,combs_number_all,order_all_to_one):
    g = []
    for i in range(combs_number_all):
        g_i=[]
        for k in dict_users_in[i]:
            index=order_all_to_one[k][i]
            g_part_1=np.log2(1+alpha_comm[k][index])-alpha_comm[k][index]
            g_part_2=2*np.sqrt(1+alpha_comm[k][index])*np.real(np.conj(beta_comm[k][index].T)*Hk_mul_Pi(H,p_c,k,i))
            g_part_3=(abs(beta_comm[k][index])**2)*T_comm[k][index]
            g_i.append(g_part_1+g_part_2-g_part_3)
        g.append(g_i)
    return g

def count_g_p(P_p,r_p,n_p,N_users,T_pri):
    g=[]
    for k in range(N_users):
        g.append(r_p[k]+np.real(2*np.conj(n_p[k].T)@P_p[:,k][:, np.newaxis])[0][0]-np.real(T_pri[k]))
    return g

def count_g_c(P_c,r_c,n_c,T_c,dict_users_in,combs_number_all,order_all_to_one):
    g = []
    for i in range(combs_number_all):
        g_i=[]
        for k in dict_users_in[i]:
            index=order_all_to_one[k][i]
            g_i.append(r_c[k][index] + np.real(2 * np.conj(n_c[k][index].T) @ P_c[:, i][:, np.newaxis])[0][0] - np.real(T_c[k][index]))
        g.append(g_i)
    return g

def quad_form(P,m,i,k):
    sum=((np.conj(P[:, i][:, np.newaxis].T)@m[k])@P[:, i][:, np.newaxis])[0][0]
    return sum
def quad_form_c(P,m_c,i,k,j):
    sum=((np.conj(P[:, i][:, np.newaxis].T)@m_c[k][j])@P[:, i][:, np.newaxis])[0][0]
    return sum

def count_beta_mul_T_avg(m_p,m_c, P, P_c, N_users,m,order_list):
    T_p = []
    T_c = []
    for k in range(N_users):
        T_private_k = 0
        T_c_k = [0 for j in range(m)]
        for i in range(N_users):
            T_private_k += quad_form(P, m_p, i, k)
        for i in order_list[k][0]:
            T_private_k += quad_form(P_c, m_p, i, k)
        T_p.append(T_private_k)
        for i in range(N_users):
            for j in range(m):
                T_c_k[j]+=quad_form_c(P,m_c,i,k,j)
        for i in order_list[k][0]:
            for j in range(m):
                T_c_k[j]+=quad_form_c(P_c,m_c,i,k,j)


        order_list_k_reverse = reversed(order_list[k][1])
        iter = 1
        for i in order_list_k_reverse:
            for j in range(m+1-iter):
                T_c_k[j] += quad_form_c(P_c,m_c,i,k,j)
            iter+=1
        T_c.append(T_c_k)
    return T_p, T_c

def update_lambda(g_comm,rou,lambda_i,combs_number_all,order_list,dict_users_in):
    lambda_new=[[0 for j in range(len(lambda_i[i]))] for i in range(len(lambda_i))]
    y,index=count_y(g_comm,combs_number_all,dict_users_in)
    summ_list=[]
    for j in range(combs_number_all):
        summ=0
        list_j=dict_users_in[j]
        for k in range(len(list_j)):
            i=list_j[k]
            if y[j] >= 0:
                lambda_new[j][i] = ((y[j] + rou) / (g_comm[j][k] + rou)) * lambda_i[j][i]
                summ+=lambda_i[j][i]- lambda_new[j][i]
            else:
                lambda_new[j][i] = (rou - y[j]) / (g_comm[j][k] - 2 * y[j] + rou) * lambda_i[j][i]
                summ+=lambda_i[j][i]- lambda_new[j][i]
        lambda_new[j][index[j]] = lambda_i[j][index[j]] + summ
        summ_list.append(abs(summ))
    return lambda_new,summ_list

def init_P(H,Pt,Nr,combination,alpha):
    P_common=[]
    P_private=[]
    for comb_1 in range(len(combination)):
        if comb_1 == 0:
            power = Pt * alpha[comb_1]
        else:
            power = Pt * alpha[comb_1]/len(combination[comb_1])
        for comb_2 in combination[comb_1]:
            a,b,c=np.linalg.svd(H[:,comb_2])
            P_common.append(np.sqrt(power) * a[:, 0][:, np.newaxis])
    power_pri=Pt*alpha[-1]/Nr
    for i in range(Nr):
        P_private.append(H[:,i][:,np.newaxis ]/np.linalg.norm(H[:,i])*np.sqrt(power_pri))
    return np.concatenate(P_common,axis=1),np.concatenate(P_private,axis=1)

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

def lambda_init(H,combinations,combs_number_all,comb_dict_org,N_users):
    lambda_output_g = [[0 for j in range(N_users)] for i in range(combs_number_all)]
    for comb_layer in range(len(combinations)):
        for comb_i in range(len(combinations[comb_layer])):
            comb=combinations[comb_layer][comb_i]
            comb_j=comb_dict_org[tuple(comb)]
            lambda_i = found_order(H[:,comb], len(comb))
            lambda_i = [abs(l_i) ** 2 for l_i in lambda_i]
            sum_lambda = sum(lambda_i)
            lambda_i = [l_i / sum_lambda for l_i in lambda_i]
            i=0
            for k in comb:
                lambda_output_g[comb_j][k]=lambda_i[i]
                i+=1
    return lambda_output_g

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

def count_comm_rate(T,I,combs_number_all,dict_users_in,order_all_to_one):
    rate=[]
    for i in range(combs_number_all):
        rate_i=[]
        for k in dict_users_in[i]:
            index=order_all_to_one[k][i]
            rate_i.append(np.log2(T[k][index] / I[k][index]))
        rate.append(rate_i)
    return rate

def produce_combination(N_users):
    comm_combine=[]
    for i in range(N_users):
        comm_combine.append(i)
    combination=[]
    combination.append([comm_combine])
    for i in range(N_users-3,-1,-1):
        combination.append(list(combinations(comm_combine, i+2)))
    for combs in combination:
        for i in range(len(combs)):
            combs[i]=list(combs[i])
    return combination

def produce_combination_dict(combination,N_users):
    numb=0
    numbs=[0]
    d=dict()
    d2=dict()
    d3=dict()
    count=0
    for combs in combination:
        numb+=len(combs)
        numbs.append(numb)
        for i in range(len(combs)):
            l=[j for j in range(N_users) if j not in combs[i]]
            comb_tuple=tuple(combs[i])
            d3[comb_tuple]=count
            d[count]=tuple(l)
            d2[count]=comb_tuple
            count+=1
    return d,d2,d3,numbs



def comb_to_string(comb):
    s=''
    for i in comb:
        s+=str(i+1)
    return s

def string_to_comb(input):
    comb=[]
    for s in input:
        number=int(s)-1
        comb.append(number)
    return comb

def produce_order_for_every_user(comb_dict,combinations,N_users,combs_number_all):
    list=[]
    layer_number_list=[]
    order_all_to_one=[[-1 for i in range(combs_number_all)] for j in range(N_users)]
    for k in range(N_users):
        k_list=[]
        k_not_in = []
        k_in = []
        i=0
        for comb_layer in range(0,len(combinations)):
            for comb in combinations[comb_layer]:
                if k in comb:
                    index=comb_dict[tuple(comb)]
                    k_in.append(index)
                    order_all_to_one[k][index]=i
                    i += 1
                else:
                    k_not_in.append(comb_dict[tuple(comb)])
            if k==0:
                layer_number_list.append(len(k_in))
        k_list.append(k_not_in)
        k_list.append(k_in)
        list.append(k_list)
    return list,layer_number_list,order_all_to_one
import time

def count_avg_alpha_beta_one(H,P,P_c,N_users, order_list):
    T_pri, T_comm = count_T(H, P, P_c, N_users, order_list)
    I_pri, I_comm = count_I(T_pri, T_comm, H, P, N_users)
    alpha_comm, alpha_pri = count_SINR(I_comm, I_pri, N_users, P_c, P, H, order_list)
    beta_comm, beta_pri = count_beta(alpha_comm, alpha_pri, N_users, P_c, P, H, T_comm, T_pri,
                                     order_list)
    return alpha_comm, beta_comm,alpha_pri,beta_pri


def count_first_part_of_update_P_c(i,lambda_i,m_c,m_p,mu,Nt,order_list,dict_users_not_in,dict_users_in):
    summ=0
    for j in dict_users_not_in[i]:
        summ+=m_p[j]
    for m in dict_users_in[i]:
        order_list_m=order_list[m][1]
        index = order_list_m.index(i)
        for j in range(index+1):
            summ+=m_c[m][j]*lambda_i[order_list[m][1][j]][m]
    for m in dict_users_not_in[i]:
        for j in range(len(order_list[m][1])):
            summ+=m_c[m][j]*lambda_i[order_list[m][1][j]][m]
    return summ+mu*np.eye(Nt)

def count_second_part_of_update_P_c(i,n_c,lambda_i,dict_users_in,order_list):
    summ=0
    for k in dict_users_in[i]:
        index = order_list[k][1].index(i)
        summ+=n_c[k][index]*lambda_i[i][k]
    return summ

def update_P_c(m_c,n_c,m_p,lambda_i,mu,combs_number_all,Nt,order_list,dict_users_not_in,dict_users_in):
    P_c=[]
    for i in range(combs_number_all):
        part_1=count_first_part_of_update_P_c(i,lambda_i,m_c,m_p,mu,Nt,order_list,dict_users_not_in,dict_users_in)
        part_2=count_second_part_of_update_P_c(i,n_c,lambda_i,dict_users_in,order_list)
        P_c.append(np.linalg.inv(part_1)@part_2)
    return np.concatenate(P_c,axis=1)
def count_first_part_of_update_P_p(lambda_i,m_p,m_c,mu,N_users,Nt,order_list):
    summ=0
    for k in range(N_users):
        first_part=m_p[k]
        second_part=0
        for i in range(len(order_list[k][1])):
            second_part+=(lambda_i[order_list[k][1][i]][k])*m_c[k][i]
        summ+=first_part+second_part
    return summ+mu*np.eye(Nt)
def update_P_p(m_p,m_c,n_p,lambda_i,N_users,Nt,mu,order_list):
    P=[]
    for k in range(N_users):
        part_1 = count_first_part_of_update_P_p(lambda_i,m_p,m_c,mu,N_users,Nt,order_list)
        part_2 = n_p[k]
        P.append(np.linalg.inv(part_1)@part_2)
    return np.concatenate(P,axis=1)

def count_m_n_r(H_reals,numbers_of_H_real,P,P_c,N_users,Nt, order_list,m):
    m_p_s,m_c_s=[np.zeros((Nt,Nt)) for k in range(N_users)],[[np.zeros((Nt,Nt)) for i in range(m)]for k in range(N_users)]
    n_p_s,n_c_s=[np.zeros((Nt,1)) for k in range(N_users)],[[np.zeros((Nt,1)) for i in range(m)]for k in range(N_users)]
    r_c_s,r_p_s=[[0 for i in range(m)]for k in range(N_users)],[0 for k in range(N_users)]
    for i in range(numbers_of_H_real):
        H_real=H_reals[i]
        alpha_comm, beta_comm,alpha_pri,beta_pri=count_avg_alpha_beta_one(H_real,P,P_c,N_users, order_list)
        beta_pri_square=[abs(beta_pri[k])**2 for k in range(N_users)]
        beta_comm_square=[[abs(beta_comm[k][i])**2 for i in range(m)]for k in range(N_users)]
        square_of_H=[square_of_Hk(H_real,k) for k in range(N_users)]
        m_p=[beta_pri_square[k]*square_of_H[k] for k in range(N_users)]
        n_p=[np.sqrt(1+alpha_pri[k])*beta_pri[k]*H_real[:, k][:, np.newaxis] for k in range(N_users)]
        m_c=[[beta_comm_square[k][i]*square_of_H[k] for i in range(m)] for k in range(N_users)]
        n_c=[[np.sqrt(1+alpha_comm[k][i])*beta_comm[k][i]*H_real[:, k][:, np.newaxis] for i in range(m)] for k in range(N_users)]
        r_c=[[np.log2(1+alpha_comm[k][i])-beta_comm_square[k][i]-alpha_comm[k][i] for i in range(m)] for k in range(N_users)]
        r_p=[np.log2(1+alpha_pri[k])-beta_pri_square[k]-alpha_pri[k] for k in range(N_users)]
        m_p_s=[m_p_s[k]+m_p[k] for k in range(N_users)]
        m_c_s=[[m_c_s[k][i]+m_c[k][i] for i in range(m)]for k in range(N_users)]
        n_p_s=[n_p_s[k]+n_p[k] for k in range(N_users)]
        n_c_s=[[n_c_s[k][i]+n_c[k][i]for i in range(m)] for k in range(N_users)]
        r_c_s=[[r_c_s[k][i]+r_c[k][i]for i in range(m)] for k in range(N_users)]
        r_p_s=[r_p_s[k]+r_p[k] for k in range(N_users)]
    m_p_avg = [m_p_s[k]/numbers_of_H_real for k in range(N_users)]
    m_c_avg = [[m_c_s[k][i]/numbers_of_H_real for i in range(m)] for k in range(N_users)]
    n_p_avg= [n_p_s[k]/numbers_of_H_real for k in range(N_users)]
    n_c_avg = [[n_c_s[k][i]/numbers_of_H_real for i in range(m)]  for k in range(N_users)]
    r_c_avg= [[r_c_s[k][i]/numbers_of_H_real for i in range(m)] for k in range(N_users)]
    r_p_avg = [r_p_s[k]/numbers_of_H_real for k in range(N_users)]
    return m_p_avg,m_c_avg,n_p_avg,n_c_avg,r_c_avg,r_p_avg
def order(H_estimate,H_reals,Nt,N_users,rho,tolerance,Pt,alpha,combinations,tolerance_inner,
          combs_number_all,comb_dict_org,maxcount,dict_users_not_in,dict_users_in,order_list,order_all_to_one):
    flag=1
    count=0
    obj_past=0
    mu=10/Pt
    m=len(order_list[0][1])
    P_c,P=init_P(H_estimate,Pt,N_users,combinations,alpha)
    lambda_i=lambda_init(H_estimate,combinations,combs_number_all,comb_dict_org,N_users)
    numbers_of_H_real=len(H_reals)
    T1=time.time()
    while(flag):
        m_p,m_c,n_p,n_c,r_c,r_p=count_m_n_r(H_reals, numbers_of_H_real, P, P_c, N_users, Nt, order_list, m)
        count_inner = 0
        while (count_inner <= maxcount):
            P_c=update_P_c(m_c,n_c,m_p,lambda_i,mu,combs_number_all,Nt,order_list,dict_users_not_in,dict_users_in)
            P=update_P_p(m_p,m_c,n_p,lambda_i,N_users,Nt,mu,order_list)
            beta_square_mul_T_p,beta_square_mul_T_c=count_beta_mul_T_avg(m_p,m_c, P, P_c, N_users,m,order_list)
            g_comm = count_g_c(P_c, r_c, n_c, beta_square_mul_T_c, dict_users_in, combs_number_all, order_all_to_one)
            lambda_new,summ=update_lambda(g_comm,rho,lambda_i,combs_number_all,order_list,dict_users_in)
            mu_new = (np.trace(P@np.conj(P.T)+ P_c@np.conj(P_c.T)  ) ) / Pt  * mu
            e=count_e(mu,mu_new,summ)
            lambda_i=lambda_new
            mu=mu_new
            if np.linalg.norm(e)<tolerance_inner:
                break
            count_inner=count_inner+1
        beta_square_mul_T_p, beta_square_mul_T_c = count_beta_mul_T_avg(m_p, m_c, P, P_c, N_users, m,order_list)
        g_comm = count_g_c(P_c,r_c,n_c,beta_square_mul_T_c,dict_users_in,combs_number_all,order_all_to_one)
        g_pri=count_g_p(P,r_p,n_p,N_users,beta_square_mul_T_p)
        obj=sum(g_pri)
        for i in range(combs_number_all):
            obj+=min(g_comm[i])
        if abs(obj - obj_past) <= tolerance:
            flag = 0
        else:
            obj_past = obj
            count = count + 1
        if count >= 1000:
            break
    rates_pri,rates_comm=count_avg_rate(H_reals, numbers_of_H_real, P, P_c, N_users, combs_number_all, order_list, combs_number_all, dict_users_in, order_all_to_one)
    obj = sum(rates_pri)
    for k in range(combs_number_all):
        obj+=min(rates_comm[k])
    T2=time.time()
    return obj,T2-T1

def produce_para_of_comb(combinations,N_users):
    dict_users_not_in,dict_users_in,comb_dict_org,numbers=produce_combination_dict(combinations,N_users)
    combs_number_all=numbers[-1]
    order_list,layer_number_list,order_all_to_one=produce_order_for_every_user(comb_dict_org, combinations, N_users,combs_number_all)
    return combs_number_all, comb_dict_org,dict_users_not_in, dict_users_in, order_list, order_all_to_one
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

def count_avg_rate(H_reals, numbers_of_H_real, P, P_c, N_users,total, order_list, combs_number_all, dict_users_in, order_all_to_one):
    T_pri, T_comm = count_T(H_reals[0], P, P_c, N_users, order_list)
    I_pri, I_comm = count_I(T_pri, T_comm, H_reals[0], P, N_users)
    rates_c_t = count_comm_rate(T_comm, I_comm, combs_number_all, dict_users_in, order_all_to_one)
    rates_p_t = count_pri_rate(T_pri, I_pri, N_users)
    for i in range(1,numbers_of_H_real):
        H_real=H_reals[i]
        T_pri, T_comm = count_T(H_real, P, P_c, N_users, order_list)
        I_pri, I_comm = count_I(T_pri, T_comm, H_real, P, N_users)
        rates_c = count_comm_rate(T_comm, I_comm, combs_number_all, dict_users_in, order_all_to_one)
        rates_p = count_pri_rate(T_pri, I_pri, N_users)
        for k in range(N_users):
            rates_p_t[k]+=rates_p[k]
        for i in range(total):
            for k in range(len(rates_c_t[i])):
                rates_c_t[i][k]+=rates_c[i][k]
    rates_p_avg = [rates_p_t[k]/numbers_of_H_real for k in range(N_users)]
    rates_c_avg = [[rates_c_t[i][k]/numbers_of_H_real for k in range(len(rates_c_t[i]))] for i in range(total)]
    return rates_p_avg,rates_c_avg

if __name__ == "__main__":
    print('系统:', platform.system())
    T1 = time.time()
    Nt=4
    N_users=4
    rho=2
    tolerance=1e-3
    transmit_SNr=20
    Pt = 10 ** (transmit_SNr / 10)
    maxcount=1000
    combinations=produce_combination(N_users)
    inner_tolerance=10 **(- 5)
    alpha=[0.6,0.2,0.1,0.1]
    bias=[1,1,1,1]
    alpha_i = 0.6
    P_e = Pt ** (-alpha_i)
    N_random_H=1000
    H_estimate,H_reals,H_errors=produce_H(1, N_users, Nt, bias, P_e, N_random_H)
    combs_number_all, comb_dict_org, dict_users_not_in, dict_users_in, order_list, order_all_to_one=produce_para_of_comb(combinations,N_users)
    SR=order(H_estimate,H_reals,Nt, N_users, rho, tolerance, Pt, alpha, combinations,inner_tolerance ,
          combs_number_all, comb_dict_org, maxcount, dict_users_not_in, dict_users_in, order_list, order_all_to_one)
    print(SR)
    T2 =time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
