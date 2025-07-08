import multiprocessing
from itertools import permutations
import numpy as np
import alg.MULP_cvx
import alg.SCSIC_cvx
import alg.SCSICpergroup_cvx
import alg.GRS_cvx
import alg.RS1layer_cvx
import alg.HRS_cvx
import alg.RS1layer
import alg.HRS
import alg.GRS
import alg.SCSIC
import alg.SCSICpergroup
import alg.MULP

def adjust_H(perm,Nt,N_users,H_estimate,H_reals):
    H_perm = np.ones((Nt, N_users)) + 1j * np.zeros((Nt, N_users))
    length = len(perm)
    for j in range(length):
        H_perm[:, j] = H_estimate[:, perm[j]]
    H_reals_perm = []
    for h in H_reals:
        h_perm = np.ones((Nt, N_users)) + 1j * np.zeros((Nt, N_users))
        for j in range(length):
            h_perm[:, j] = h[:, perm[j]]
        H_reals_perm.append(h_perm)
    return H_perm,H_reals_perm

def RSMA_onelayer(H_estimate,H_reals,N_users,tolerance,rho,Pt,Nt,tolerance_inner,alpha,maxcount,weight):
    cvxSR,cvxt=alg.RS1layer_cvx.order_g(weight, H_estimate, H_reals, tolerance, Pt, N_users, Nt, 0, alpha)
    fpSR,fpt=alg.RS1layer.order(H_estimate, H_reals, Nt, N_users, rho, tolerance, Pt, alpha, tolerance_inner, maxcount)
    return fpSR,fpt,cvxSR,cvxt

def HRS_1(group,H_estimate,H_reals,perm,N_users,Nt,tolerance,alpha_inner,Pt,rho,alphas,tolerance_inner,maxcount,weight):
    fpSR, fpt=0,0
    cvxSR, cvxt=0,0
    try:
        cvxSR,cvxt=alg.HRS_cvx.order_g(H_estimate, H_reals, group, weight, tolerance, Pt, N_users, Nt, 0, alphas, alpha_inner)
        fpSR,fpt=alg.HRS.order(H_estimate, H_reals, group, Nt, N_users, rho, tolerance, Pt, alphas, alpha_inner, tolerance_inner, maxcount)
    except:
        print("this order"+perm+"is broken")
    return fpSR,fpt,cvxSR,cvxt

def HRS_full_perm(group,H_estimate,H_reals,N_users,Nt,tolerance,alpha_inner,Pt,rho,alphas,tolerance_inner,maxcount,weight):
    pool = multiprocessing.Pool(processes=3)
    if N_users ==4:
        perms=[[1,0,2,3],[2,1,3,0],[2,0,3,1]]
    elif N_users==3:
        perms = [[2, 0,1],[1, 0, 2], [2, 1,  0]]
    args_list=[]
    for perm in perms:
        H_e_perm,H_reals_perm=adjust_H(perm, Nt, N_users, H_estimate, H_reals)
        args_list.append((group,H_e_perm,H_reals_perm,perm,N_users,Nt,tolerance,alpha_inner,Pt,rho,alphas,tolerance_inner,maxcount,weight))
    answer= pool.starmap(HRS_1, args_list)
    fpsum_rates=[answer[i][0] for i in range(len(answer))]
    fpT_s=[answer[i][1] for i in range(len(answer))]
    cvxsum_rates=[answer[i][2] for i in range(len(answer))]
    cvxT_s=[answer[i][3] for i in range(len(answer))]
    fpSR,cvxSR = np.max(fpsum_rates),np.max(cvxsum_rates)
    fpT,cvxT=np.sum(fpT_s),np.sum(cvxT_s)
    return fpSR,fpT,cvxSR,cvxT

def MULP(H_estimate, H_reals,N_users,Nt,tolerance,rho,Pt,tolerance_inner,alpha,maxcount,weight):
    cvxSR,cvxt= alg.MULP_cvx.order_g(H_reals, H_estimate, weight, tolerance, Pt, N_users, Nt, 0, alpha)
    fpSR,fpt= alg.MULP.order(H_estimate, H_reals, Nt, N_users, rho, tolerance, Pt, alpha, tolerance_inner, maxcount)
    return fpSR,fpt,cvxSR,cvxt

def NOMA(H_estimate,H_reals,perm,N_users,Nt,tolerance,rho,Pt,tolerance_inner,maxcount,alpha,weight):
    fpSR, fpt=0,0
    cvxSR, cvxt=0,0
    try:
        cvxSR,cvxt=alg.SCSIC_cvx.order_g(weight, H_estimate, H_reals, tolerance, Pt, N_users, Nt, 0, alpha)
        fpSR,fpt=alg.SCSIC.order(H_estimate, H_reals, Nt, N_users, rho, tolerance, Pt, alpha, tolerance_inner, maxcount)
    except:
        print("this order"+perm+"is broken")
    return fpSR,fpt,cvxSR,cvxt

def NOMA_full_perm(pro_numb,H_estimate,H_reals,N_users,Nt,tolerance,rho,Pt,tolerance_inner,maxcount,alpha,weight):
    pool = multiprocessing.Pool(processes=pro_numb)
    combines = [i for i in range(N_users)]
    perm=list(permutations(combines,N_users))
    perms=[list(i) for i in perm]
    args_list=[]
    for perm in perms:
        H_e_perm,H_reals_perm=adjust_H(perm, Nt, N_users, H_estimate, H_reals)
        args_list.append((H_e_perm,H_reals_perm,perm,N_users,Nt,tolerance,rho,Pt,tolerance_inner,maxcount,alpha,weight))
    answer= pool.starmap(NOMA, args_list)
    fpsum_rates=[answer[i][0] for i in range(len(answer))]
    fpT_s=[answer[i][1] for i in range(len(answer))]
    cvxsum_rates=[answer[i][2] for i in range(len(answer))]
    cvxT_s=[answer[i][3] for i in range(len(answer))]
    fpSR,cvxSR = np.max(fpsum_rates),np.max(cvxsum_rates)
    fpT,cvxT=np.sum(fpT_s),np.sum(cvxT_s)
    return fpSR,fpT,cvxSR,cvxT

def NOMA_group(H_estimate,H_reals,group,perm,N_users,Nt,tolerance,rho,Pt,tolerance_inner,maxcount,alpha,weight):
    fpSR, fpt=0,0
    cvxSR, cvxt=0,0
    try:
        cvxSR,cvxt = alg.SCSICpergroup_cvx.order_g(H_estimate, H_reals, group, weight, tolerance, Pt, N_users, Nt, 0, alpha)
        fpSR,fpt = alg.SCSICpergroup.order(H_estimate, H_reals, group, Nt, N_users, rho, tolerance, Pt, alpha, tolerance_inner, maxcount)
    except:
        print("this order"+perm+"is broken")
    return fpSR,fpt,cvxSR,cvxt

def NOMA_group_full_perm(pro_numb,H_estimate,H_reals,N_users,Nt,tolerance,rho,Pt,tolerance_inner,maxcount,alpha,group,weight):
    pool = multiprocessing.Pool(processes=pro_numb)
    combines = [i for i in range(N_users)]
    if N_users != 4:
        perm = list(permutations(combines, N_users))
        perms = [list(i) for i in perm]
    elif N_users == 4:
        perms = [[1, 0, 2, 3], [0, 1, 2, 3], [1, 0, 3, 2], [0, 1, 3, 2],
             [2, 1, 3, 0], [1, 2, 3, 0], [2, 1, 0, 3], [1, 2, 0, 3],
             [2, 0, 3, 1], [0, 2, 3, 1], [2, 0, 1, 3], [0, 2, 1, 3]]
    args_list=[]
    for perm in perms:
        H_e_perm,H_reals_perm=adjust_H(perm, Nt, N_users, H_estimate, H_reals)
        args_list.append((H_e_perm,H_reals_perm,group,perm,N_users,Nt,tolerance,rho,Pt,tolerance_inner,maxcount,alpha,weight))
    answer= pool.starmap(NOMA_group, args_list)
    fpsum_rates=[answer[i][0] for i in range(len(answer))]
    fpT_s=[answer[i][1] for i in range(len(answer))]
    cvxsum_rates=[answer[i][2] for i in range(len(answer))]
    cvxT_s=[answer[i][3] for i in range(len(answer))]
    fpSR,cvxSR = np.max(fpsum_rates),np.max(cvxsum_rates)
    fpT,cvxT=np.sum(fpT_s),np.sum(cvxT_s)
    return fpSR,fpT,cvxSR,cvxT

def GRS_1(H_estimate,H_reals,perm,Nt,  tolerance, Pt, rho, N_users,combinations, tolerance_inner,
          combs_number_all,comb_dict_org,maxcount,dict_users_not_in,dict_users_in,order_list,order_all_to_one,alpha,
                 weight,total,one,number_order,numbers,combinations_dict):
    fpSR, fpt=0,0
    cvxSR, cvxt=0,0
    try:
        cvxSR,cvxt=alg.GRS_cvx.order_g(H_estimate, H_reals, Nt, weight, tolerance, Pt, total, one, number_order, N_users, combinations, numbers, combinations_dict, 0, alpha)
        fpSR,fpt=alg.GRS.order(H_estimate, H_reals, Nt, N_users, rho, tolerance, Pt, alpha, combinations, tolerance_inner,
                               combs_number_all, comb_dict_org, maxcount, dict_users_not_in, dict_users_in, order_list, order_all_to_one)
    except:
        print("this order failed",perm)
    return fpSR,fpt,cvxSR,cvxt

def GRS_full_perm(pro_numb,H_estimate,H_reals,N_users,Nt,tolerance,rho,Pt,tolerance_inner,maxcount,alpha,weight):
    combinations = alg.GRS.produce_combination(N_users)
    combs_number_all, comb_dict_org, dict_users_not_in, dict_users_in, order_list, order_all_to_one = alg.GRS.produce_para_of_comb(
        combinations, N_users)
    combinations_dict,numbers=alg.GRS_cvx.produce_combination_dict(combinations)
    number_order=alg.GRS_cvx.analysis_order(alg.GRS_cvx.produce_order(N_users, combinations), combinations_dict)
    pool = multiprocessing.Pool(processes=pro_numb)
    combines = [i for i in range(N_users)]
    perm=list(permutations(combines,N_users))
    perms=[list(i) for i in perm]
    args_list=[]
    for perm in perms:
        H_e_perm,H_reals_perm=adjust_H(perm, Nt, N_users, H_estimate, H_reals)
        args_list.append((H_e_perm,H_reals_perm,perm,Nt,  tolerance, Pt, rho, N_users,combinations, tolerance_inner,
          combs_number_all,comb_dict_org,maxcount,dict_users_not_in,dict_users_in,order_list,order_all_to_one,alpha,
                 weight,numbers[-1] + N_users, numbers[-1],number_order,numbers,combinations_dict))
    answer= pool.starmap(GRS_1, args_list)
    fpsum_rates=[answer[i][0] for i in range(len(answer))]
    fpT_s=[answer[i][1] for i in range(len(answer))]
    cvxsum_rates=[answer[i][2] for i in range(len(answer))]
    cvxT_s=[answer[i][3] for i in range(len(answer))]
    fpSR,cvxSR = np.max(fpsum_rates),np.max(cvxsum_rates)
    fpT,cvxT=np.sum(fpT_s),np.sum(cvxT_s)
    return fpSR,fpT,cvxSR,cvxT

