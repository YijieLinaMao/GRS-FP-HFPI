
import cvxpy as cvx
from pylab import *
from itertools import combinations,permutations
import multiprocessing
import copy

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

def Init_P_g(H,Pt,N_users,combination,alpha):
    P_common=[]
    P_private=[]
    for comb_1 in range(len(combination)):
        if comb_1 == 0:
            power = Pt * alpha[comb_1]
        else:
            power = Pt * alpha[comb_1]/len(combination[comb_1])
        for comb_2 in combination[comb_1]:
            a,b,c=np.linalg.svd(H[:,comb_2])
            P_common.append(np.sqrt(power)*a[:,0][:,np.newaxis ])
    power_pri=Pt*alpha[-1]/N_users
    for i in range(N_users):
        P_private.append(H[:,i][:,np.newaxis ]/np.linalg.norm(H[:,i])*np.sqrt(power_pri))
    P_total=P_common+P_private
    return np.concatenate(P_total,axis=1)

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

def produce_combination_dict(combination):
    comb_dict=[]
    numb=0
    numbs=[0]
    for combs in combination:
        d = dict()
        numb+=len(combs)
        numbs.append(numb)
        for i in range(len(combs)):
            s=''
            for j in combs[i]:
                s=s+str(j+1)
            d[s]=i
        comb_dict.append(d)
    return comb_dict,numbs

def analysis_order(orders,dictionary):
    common_order=[]
    for i in range(len(orders)):
        order_1 = []
        for comm in orders[i]:
            order_2 = []
            for com_1 in comm:
                order_2.append(dictionary[i][com_1])
            order_1.append(order_2)
        common_order.append(order_1)
    return common_order

def count_T_MMSE_g(H,P,number_order,N_users,combination,numbers):
    T=[]
    for k in range(N_users):
        T_k=[]
        T_private_k=1
        for i in range(N_users):
            T_private_k+=abs_square_of_Hk_mul_Pi(H,P,k,numbers[-1]+i)
        T_comm_k = T_private_k
        for i in range(len(number_order)-1,0,-1):
            for j in range(len(combination[i])):
                if j not in number_order[i][k]:
                    T_comm_k+=abs_square_of_Hk_mul_Pi(H,P,k,numbers[i]+j)
        T_k.append(T_comm_k)
        for i in range(len(number_order)-1,0,-1):
            for ii in range(len(number_order[i][k])-1,-1,-1):
                T_comm_k+=abs_square_of_Hk_mul_Pi(H,P,k,numbers[i]+number_order[i][k][ii])
                T_k.append(T_comm_k)
        T_comm_k+=abs_square_of_Hk_mul_Pi(H,P,k,0)
        T_k.append(T_comm_k)
        T_k.reverse()
        T.append(T_k)
    return T

def count_I_MMSE_g(T,H,P,N_users,numbers):
    I=[]
    for k in range(N_users):
        I_k=T[k][1:len(T[k])+1]
        I_k_k=I_k[-1]-abs_square_of_Hk_mul_Pi(H,P,k,numbers[-1]+k)
        I_k.append(I_k_k)
        I.append(I_k)
    return I

def count_g_MMSE_g(H,P,T,N_users,numbers,number_order):
    g=[]
    for k in range(N_users):
        g_k=[]
        g_k.append(Pi_mul_Hk(H,P,k,0)*(T[k][0]**(-1)))
        numb = 1
        for i in range(1,len(number_order)):
            for ii in range(len(number_order[i][k])):
                PH=Pi_mul_Hk(H,P,k,numbers[i]+number_order[i][k][ii])
                g_k.append(count_g_i(PH,T[k][numb+ii]))
            numb+=len(number_order[i][0])
        g_k.append(count_g_i(Pi_mul_Hk(H,P,k,numbers[-1]+k),T[k][-1]))
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

def count_ksai_tot_g(ksai,x,N_users,m):
    ksai_tot=[]
    for k in range(N_users):
        ksai_tot_k=ksai[k][-1]
        for j in range(m):
            ksai_tot_k+=x[k][j]
        ksai_tot.append(ksai_tot_k)
    return ksai_tot

def find_index(order,numbers_layer,numb):
    d={}
    for i in range(len(numbers_layer)):
        for j in range(len(numbers_layer[i])):
            if numbers_layer[i][j]==order:
                d[i]=j+numb
    return d

def ksai_comm_constraint(combination,x,ksai,dicitonary,number_order):
    numb=0
    constraints=[]
    for layer in range(len(combination)):
        for i in range(len(combination[layer])):
            L = []
            s = ''
            for ii in combination[layer][i]:
                s += str(ii+1)
            order=dicitonary[layer][s]
            d=find_index(order,number_order[layer],numb)
            left_equation=1
            for key, value in d.items():
                L.append(ksai[key][value])
                left_equation+=x[key][value]
            for l in L:
                constraints.append(left_equation >= l)
        numb += len(number_order[layer][0])
    return constraints

def count_eplison_cvx_g(T,g,H,P,N_users,number_order,numbers):
    eplison=[]
    for k in range(N_users):
        eplison_k=[]
        eplison_k.append(count_e_cvx(g[k][0],H,P,k,0,T[k][0]))
        numb = 1
        for i in range(1,len(number_order)):
            for ii in range(len(number_order[i][k])):
                eplison_k.append(count_e_cvx(g[k][numb+ii],H,P,k,numbers[i]+number_order[i][k][ii],T[k][numb+ii]))
            numb += len(number_order[i][0])
        eplison_k.append(count_e_cvx(g[k][-1],H,P,k,numbers[-1]+k,T[k][-1]))
        eplison.append(eplison_k)
    return eplison


def quad_form(P,phi,i,k,j):
    return cvx.quad_form(P[:, i][:, np.newaxis], phi[k][j])

def count_ksai(ksai,t,f,P,v,u,i,k,j):
    return ksai+t[k][j]-2*cvx.real(np.conj(f[k][j].T)@P[:,i][:, np.newaxis])-v[k][j]+u[k][j]

def count_ksai_imp(u,t,phi,f,v,P_opt,N_users,m,numbers,number_order,combination):
    ksai=[]
    for k in range(N_users):
        ksai_k=[0 for i in range(m+1)]
        for i in range(N_users):
            for m_j in range(m+1):
                ksai_k[m_j]+=quad_form(P_opt,phi,numbers[-1]+i,k,m_j)
        for i in range(len(number_order)-1,0,-1):
            for j in range(len(combination[i])):
                if j not in number_order[i][k]:
                    for m_j in range(m+1):
                        ksai_k[m_j]+=quad_form(P_opt,phi,numbers[i]+j,k,m_j)
        iter=1
        for i in range(len(number_order)-1,0,-1):
            for ii in range(len(number_order[i][k])-1,-1,-1):
                temp=numbers[i]+number_order[i][k][ii]
                for m_j in range(0,m+1-iter):
                    ksai_k[m_j]+=quad_form(P_opt,phi,temp,k,m_j)
                iter+=1
        ksai_k[0]+=quad_form(P_opt,phi,0,k,0)
        numb = 1
        for i in range(1,len(number_order)):
            for ii in range(len(number_order[i][k])):
                ksai_k[numb+ii] = count_ksai(ksai_k[numb+ii], t, f, P_opt, v, u, numbers[i]+number_order[i][k][ii], k, numb+ii)
            numb +=len(number_order[i][k])
        ksai_k[0] = count_ksai(ksai_k[0], t, f, P_opt, v, u, 0, k, 0)
        ksai_k[-1] = count_ksai(ksai_k[-1], t, f, P_opt, v, u, numbers[-1]+k, k, -1)
        ksai.append(ksai_k)
    return ksai

def optimize_P_x_g(u,t, phi, f, v,Nt,Pt,miu_weight_WSR,count,number_order,N_users,combination,numbers,rth,dictionary,m):
    P_opt = cvx.Variable((Nt, count),complex=True)
    x=cvx.Variable((N_users,m))
    ksai = count_ksai_imp(u,t,phi,f,v,P_opt,N_users,m,numbers,number_order,combination)
    ksai_tot = count_ksai_tot_g(ksai, x,N_users,m)
    sum_P=0+0j
    for i in range(count):
        sum_P+=cvx.quad_form(P_opt[:,i][:, np.newaxis],np.eye(Nt))
    constraint1 = ksai_comm_constraint(combination,x,ksai,dictionary,number_order)
    constraint2 = cvx.real(sum_P)<= Pt
    constraint3 = []
    for k in range(N_users):
        constraint3.append(ksai_tot[k] <= 1-rth)
    constraint4 = x<= np.zeros((N_users,m))
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

def count_rate_tot(rate,x,N_users,m):
    rate_tot=[]
    for k in range(N_users):
        rate_tot_k=rate[k]
        for i in range(m):
            rate_tot_k-=x[k][i]
        rate_tot.append(rate_tot_k)
    return rate_tot

def count_sum_rate(miu_weight_WSR,rate_tot,N_users):
    sum_rate=0
    for i in range(N_users):
        sum_rate+=miu_weight_WSR[i]*rate_tot[i]
    return sum_rate
import time

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
def count_one_of_avg(H_real,P,N_users,number_order,numbers,combinations):
    T = count_T_MMSE_g(H_real, P, number_order, N_users, combinations, numbers)
    I = count_I_MMSE_g(T, H_real, P, N_users, numbers)
    e = count_eplison_MMSE_g(T, I, N_users)
    g = count_g_MMSE_g(H_real, P, T, N_users, numbers, number_order)
    u = count_u_MMSE_g(e, N_users)
    t=count_t(u, g, N_users)
    phi=count_phi(t, H_real, N_users)
    f=count_f(u, H_real, g, N_users)
    v=count_v(u, N_users)
    return u,t,phi,f,v

def count_avg(number_of_H_real,H_reals,P,N_users,Nt,number_of_precoder_one_user,number_order,numbers,combinations):
    u_sum,t_sum,phi_sum,v_sum,f_sum=[],[],[],[],[]
    for k in range(N_users):
        u_sum_k,t_sum_k, phi_sum_k,v_sum_k,f_sum_k = [], [], [], [],[]
        for j in range(number_of_precoder_one_user):
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
        u,t,phi,f,v=count_one_of_avg(H_real, P, N_users,number_order,numbers,combinations)
        for k in range(N_users):
            for j in range(number_of_precoder_one_user):
                u_sum[k][j]+=u[k][j]
                t_sum[k][j]+=t[k][j]
                phi_sum[k][j]+=phi[k][j]
                f_sum[k][j]=f[k][j]+f_sum[k][j]
                v_sum[k][j]+=v[k][j]
    for k in range(N_users):
        for j in range(number_of_precoder_one_user):
            u_sum[k][j] =u_sum[k][j]/number_of_H_real
            t_sum[k][j] = t_sum[k][j]/number_of_H_real
            phi_sum[k][j] = phi_sum[k][j]/number_of_H_real
            f_sum[k][j] = f_sum[k][j]/number_of_H_real
            v_sum[k][j] = v_sum[k][j]/number_of_H_real
    return u_sum,t_sum,phi_sum,f_sum,v_sum
def count_avg_rate(number_of_H_real,H_reals,P,N_users, combinations, numbers,number_order):
    rates=[]
    for i in range(number_of_H_real):
        H_real=H_reals[i]
        T = count_T_MMSE_g(H_real, P, number_order, N_users, combinations, numbers)
        I = count_I_MMSE_g(T, H_real, P, N_users, numbers)
        rate = count_rate(T, I, N_users)
        rates.append(rate)
    avg_rates=[0 for i in range(N_users)]
    for k in range(N_users):
        avg_rates[k]+=sum([rates[i][k] for i in range(number_of_H_real)])/number_of_H_real
    return avg_rates


def order_g( H_estimate,H_reals,Nt,miu_weight_WSR,boundary,Pt,count,com_count,number_order,N_users,combinations,numbers,combinations_dict,rth,alphas):
    sum = [0]
    L=[]
    P=Init_P_g(H_estimate,Pt,N_users,combinations,alphas)
    m=0
    for i in number_order:
        m+=len(i[0])
    n = 0
    T1=time.time()
    number_of_H_real=len(H_reals)
    while True:
        n += 1
        u,t, phi, f, v=count_avg(number_of_H_real,H_reals,P,N_users,Nt,m+1,number_order,numbers,combinations)
        P,x,obj = optimize_P_x_g(u,t, phi, f, v,Nt,Pt,miu_weight_WSR,count,number_order,N_users,combinations,numbers,rth,combinations_dict,m)
        sum.append(obj)
        if abs(sum[n] - sum[n - 1]) < boundary or n>=1000:
            avg_rate=count_avg_rate(number_of_H_real,H_reals,P,N_users, combinations, numbers,number_order)
            rate_tot=count_rate_tot(avg_rate, x, N_users, m)
            sum_rate=count_sum_rate(miu_weight_WSR,rate_tot,N_users)
            L.append(sum_rate)
            break
    T2 = time.time()
    return sum_rate,T2-T1

def produce_order(N_users,combinations):
    order=[]
    for combs in range(len(combinations)):
        order_comb=[]
        for k in range(N_users):
            order_k=[]
            for combs_order in range(len(combinations[combs])):
                if k in combinations[combs][combs_order]:
                    s = ''
                    for numbers in combinations[combs][combs_order]:
                        s+=str(numbers+1)
                    order_k.append(s)
                    if combs==0:
                        break
                if combs == 0:
                    break
            order_comb.append(order_k)
        order.append(order_comb)
    return order


def initial_H_random(seed, N_users, Nt,bias):
    H=[]
    np.random.seed(seed)
    for i in range(N_users):
        h=np.random.randn(Nt)+1j * np.random.randn(Nt)
        h=np.sqrt(bias[i])*h*1/np.sqrt(2)
        H.append(np.reshape(h,(Nt,1)))
    H=np.concatenate(H,axis=1)
    return H

def RSMA_produce_all_decode_order(comb_list,combinations,layer_index):
    length = len(combinations[layer_index])
    combines = [i for i in range(length)]
    perm = list(permutations(combines, length))
    perms = [list(i) for i in perm]
    for perm in perms:
        order_for_replace = [0 for i in range(len(combinations[layer_index]))]
        for j in range(len(perm)):
            order_for_replace[perm[j]] = combinations[layer_index][j]
        combinations_new = copy.deepcopy(combinations)
        combinations_new[layer_index] = order_for_replace
        if layer_index < len(combinations) - 1:
            RSMA_produce_all_decode_order(comb_list, combinations_new, layer_index + 1)
        else:
            comb_list.append(combinations_new)
    return comb_list

def count_one_order(combination,H,N_users,Nt,tolerance,rth,Pt,weight,alpha):
    combinations_dict, numbers=produce_combination_dict(combination)
    order=produce_order(N_users, combination)
    number_order=analysis_order(order,combinations_dict)
    sum_rate = order_g(Nt, weight, H, tolerance, Pt, numbers[-1] + N_users,
                                                        numbers[-1], number_order, N_users,
                                                        combination, numbers, combinations_dict, rth, alpha)
    return sum_rate

def count_all_order(combinations,H,N_users,Nt,tolerance,rth,Pt,weight,alpha):
    combinations=combinations[0:24]
    pool = multiprocessing.Pool(processes=8)
    args_list=[]
    for comb in combinations:
        args_list.append((comb,H,N_users,Nt,tolerance,rth,Pt,weight,alpha))
    resultList = pool.starmap(count_one_order, args_list)
    pool.close()
    pool.join()
    result=max(resultList)
    return result

if __name__ == "__main__":
    T1=time.time()
    Nt = 2
    N_users = 4
    rho = 0.1
    tolerance = 1e-2
    SNR=10
    Pt = 10 ** (SNR / 10)
    weight=[1,1,1,1]
    combinations=produce_combination(N_users)
    comb_list=[]
    RSMA_produce_all_decode_order(comb_list, combinations, 1)
    combinations_dict, numbers=produce_combination_dict(combinations)
    order=produce_order(N_users, combinations)
    number_order=analysis_order(order,combinations_dict)
    alpha=[0.6,0.2,0.1,0.1]
    bias = [1, 1,1,1]
    alpha_i=1
    P_e = Pt ** (-alpha_i)
    H_estimate=initial_H_random(1, N_users, Nt, bias)*np.sqrt(1-P_e)
    H_errors=[]
    H_reals=[]
    for seed_i in range(3):
        H_error=initial_H_random(seed_i, N_users, Nt, bias)*np.sqrt(P_e)
        H_real=H_estimate+H_error
        H_errors.append(H_error)
        H_reals.append(H_real)
    SR,t=order_g( H_estimate,H_reals,Nt,weight,tolerance,Pt,numbers[-1] + N_users,numbers[-1],number_order,N_users,combinations,numbers,combinations_dict,0,alpha)
    print(SR)
    T2=time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
