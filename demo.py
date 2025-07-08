import numpy as np
import os
import algoritm
def store(filename_base_for_save):
    cvxRS1_sum_list_np = np.array(cvxRS1_sum_list)
    cvxRSMA_twolayers_sum_list_np = np.array(cvxRSMA_twolayers_list_sum)
    cvxRSMA_generalized_sum_list_np = np.array(cvxRSMA_generalized_list_sum)
    cvxNOMA_sum_list_np = np.array(cvxNOMA_sum_list)
    cvxNOMA_group_sum_list_np = np.array(cvxNOMA_group_sum_list)
    cvxMULP_list_sum_np = np.array(cvxMULP_list_sum)
    np.save(filename_base_for_save + '_cvx1RS.npy', cvxRS1_sum_list_np)
    np.save(filename_base_for_save + '_cvxHRS.npy', cvxRSMA_twolayers_sum_list_np)
    np.save(filename_base_for_save + '_cvxGRS.npy', cvxRSMA_generalized_sum_list_np)
    np.save(filename_base_for_save + '_cvxNOMA_g.npy', cvxNOMA_group_sum_list_np)
    np.save(filename_base_for_save + '_cvxNOMA.npy', cvxNOMA_sum_list_np)
    np.save(filename_base_for_save + '_cvxMULP.npy', cvxMULP_list_sum_np)
    rRS1_sum_list_np = np.array(fpRS1_sum_list)
    rRSMA_twolayers_sum_list_np = np.array(fpRSMA_twolayers_list_sum)
    rRSMA_generalized_sum_list_np = np.array(fpRSMA_generalized_list_sum)
    rNOMA_sum_list_np = np.array(fpNOMA_sum_list)
    rNOMA_group_sum_list_np = np.array(fpNOMA_group_sum_list)
    rMULP_list_sum_np = np.array(fpMULP_list_sum)
    np.save(filename_base_for_save + '_r1RS.npy', rRS1_sum_list_np)
    np.save(filename_base_for_save + '_rHRS.npy', rRSMA_twolayers_sum_list_np)
    np.save(filename_base_for_save + '_rGRS.npy', rRSMA_generalized_sum_list_np)
    np.save(filename_base_for_save + '_rNOMA_g.npy', rNOMA_group_sum_list_np)
    np.save(filename_base_for_save + '_rNOMA.npy', rNOMA_sum_list_np)
    np.save(filename_base_for_save + '_rMULP.npy', rMULP_list_sum_np)
    cvxRS1_t_list_np = np.array(cvxRS1_t_list)
    cvxRSMA_twolayers_t_list_np = np.array(cvxRSMA_twolayers_list_t)
    cvxRSMA_generalized_t_list_np = np.array(cvxRSMA_generalized_list_t)
    cvxNOMA_t_list_np = np.array(cvxNOMA_t_list)
    cvxNOMA_group_t_list_np = np.array(cvxNOMA_group_t_list)
    cvxMULP_list_t_np = np.array(cvxMULP_list_t)
    np.save(filename_base_for_save + 'cvx_1RS_t.npy', cvxRS1_t_list_np)
    np.save(filename_base_for_save + 'cvx_HRS_t.npy', cvxRSMA_twolayers_t_list_np)
    np.save(filename_base_for_save + 'cvx_GRS_t.npy', cvxRSMA_generalized_t_list_np)
    np.save(filename_base_for_save + 'cvx_NOMA_g_t.npy', cvxNOMA_group_t_list_np)
    np.save(filename_base_for_save + 'cvx_NOMA_t.npy', cvxNOMA_t_list_np)
    np.save(filename_base_for_save + 'cvx_MULP_t.npy', cvxMULP_list_t_np)
    rRS1_t_list_np = np.array(fpRS1_t_list)
    rRSMA_twolayers_t_list_np = np.array(fpRSMA_twolayers_list_t)
    rRSMA_generalized_t_list_np = np.array(fpRSMA_generalized_list_t)
    rNOMA_t_list_np = np.array(fpNOMA_t_list)
    rNOMA_group_t_list_np = np.array(fpNOMA_group_t_list)
    rMULP_list_t_np = np.array(fpMULP_list_t)
    np.save(filename_base_for_save + 'r_1RS_t.npy', rRS1_t_list_np)
    np.save(filename_base_for_save + 'r_HRS_t.npy', rRSMA_twolayers_t_list_np)
    np.save(filename_base_for_save + 'r_GRS_t.npy', rRSMA_generalized_t_list_np)
    np.save(filename_base_for_save + 'r_NOMA_g_t.npy', rNOMA_group_t_list_np)
    np.save(filename_base_for_save + 'r_NOMA_t.npy', rNOMA_t_list_np)
    np.save(filename_base_for_save + 'r_MULP_t.npy', rMULP_list_t_np)


def store_avg(filename_base_for_save):
    rRSMA_generalized_list = []
    rRSMA_onelayer_list = []
    rNOMA_list = []
    rRSMA_twolayers_list = []
    rMULP_list = []
    rNOMA_group_list = []
    cvxRSMA_generalized_list = []
    cvxRSMA_onelayer_list = []
    cvxNOMA_list = []
    cvxRSMA_twolayers_list = []
    cvxMULP_list = []
    cvxNOMA_group_list = []

    t_rRSMA_generalized_list = []
    t_rRSMA_onelayer_list = []
    t_rNOMA_list = []
    t_rRSMA_twolayers_list = []
    t_rMULP_list = []
    t_rNOMA_group_list = []
    t_cvxRSMA_generalized_list = []
    t_cvxRSMA_onelayer_list = []
    t_cvxNOMA_list = []
    t_cvxRSMA_twolayers_list = []
    t_cvxMULP_list = []
    t_cvxNOMA_group_list = []
    for i in range(compare_term_length):
        cvxRSMA_onelayer_list.append(sum(cvxRS1_sum_list[i]) / len(cvxRS1_sum_list[i]))
        cvxNOMA_group_list.append(sum(cvxNOMA_group_sum_list[i]) / len(cvxNOMA_group_sum_list[i]))
        cvxNOMA_list.append(sum(cvxNOMA_sum_list[i]) / len(cvxNOMA_sum_list[i]))
        cvxRSMA_generalized_list.append(sum(cvxRSMA_generalized_list_sum[i]) / len(cvxRSMA_generalized_list_sum[i]))
        cvxMULP_list.append(sum(cvxMULP_list_sum[i]) / len(cvxMULP_list_sum[i]))
        cvxRSMA_twolayers_list.append(sum(cvxRSMA_twolayers_list_sum[i]) / len(cvxRSMA_twolayers_list_sum[i]))
        rRSMA_onelayer_list.append(sum(fpRS1_sum_list[i]) / len(fpRS1_sum_list[i]))
        rNOMA_group_list.append(sum(fpNOMA_group_sum_list[i]) / len(fpNOMA_group_sum_list[i]))
        rNOMA_list.append(sum(fpNOMA_sum_list[i]) / len(fpNOMA_sum_list[i]))
        rRSMA_generalized_list.append(sum(fpRSMA_generalized_list_sum[i]) / len(fpRSMA_generalized_list_sum[i]))
        rMULP_list.append(sum(fpMULP_list_sum[i]) / len(fpMULP_list_sum[i]))
        rRSMA_twolayers_list.append(sum(fpRSMA_twolayers_list_sum[i]) / len(fpRSMA_twolayers_list_sum[i]))

        t_cvxRSMA_onelayer_list.append(sum(cvxRS1_t_list[i]) / len(cvxRS1_t_list[i]))
        t_cvxNOMA_group_list.append(sum(cvxNOMA_group_t_list[i]) / len(cvxNOMA_group_t_list[i]))
        t_cvxNOMA_list.append(sum(cvxNOMA_t_list[i]) / len(cvxNOMA_t_list[i]))
        t_cvxRSMA_generalized_list.append(sum(cvxRSMA_generalized_list_t[i]) / len(cvxRSMA_generalized_list_t[i]))
        t_cvxMULP_list.append(sum(cvxMULP_list_t[i]) / len(cvxMULP_list_t[i]))
        t_cvxRSMA_twolayers_list.append(sum(cvxRSMA_twolayers_list_t[i]) / len(cvxRSMA_twolayers_list_t[i]))
        t_rRSMA_onelayer_list.append(sum(fpRS1_t_list[i]) / len(fpRS1_t_list[i]))
        t_rNOMA_group_list.append(sum(fpNOMA_group_t_list[i]) / len(fpNOMA_group_t_list[i]))
        t_rNOMA_list.append(sum(fpNOMA_t_list[i]) / len(fpNOMA_t_list[i]))
        t_rRSMA_generalized_list.append(sum(fpRSMA_generalized_list_t[i]) / len(fpRSMA_generalized_list_t[i]))
        t_rMULP_list.append(sum(fpMULP_list_t[i]) / len(fpMULP_list_t[i]))
        t_rRSMA_twolayers_list.append(sum(fpRSMA_twolayers_list_t[i]) / len(fpRSMA_twolayers_list_t[i]))


    print("time:GRS:fp:"+str(t_rRSMA_generalized_list)+",GRS:cvx:"+str(t_cvxRSMA_generalized_list))
    print("time:HRS:fp:"+str(t_rRSMA_twolayers_list)+",HRS:cvx:"+str(t_cvxRSMA_twolayers_list))
    print("time:1RS:fp:"+str(t_rRSMA_onelayer_list)+",1RS:cvx:"+str(t_cvxRSMA_onelayer_list))
    print("time:MULP:fp:"+str(t_rMULP_list)+",MULP:cvx:"+str(t_cvxMULP_list))
    print("time:SC-SIC:fp:"+str(t_rNOMA_list)+",SC-SIC:cvx:"+str(t_cvxNOMA_list))
    print("time:SC-SIC per group:fp:"+str(t_rNOMA_group_list)+",SC-SIC per group:cvx:"+str(t_cvxNOMA_group_list))

    print("GRS:fp:"+str(rRSMA_generalized_list)+",GRS:cvx:"+str(cvxRSMA_generalized_list))
    print("HRS:fp:"+str(rRSMA_twolayers_list)+",HRS:cvx:"+str(cvxRSMA_twolayers_list))
    print("1RS:fp:"+str(rRSMA_onelayer_list)+",1RS:cvx:"+str(cvxRSMA_onelayer_list))
    print("MULP:fp:"+str(rMULP_list)+",MULP:cvx:"+str(cvxMULP_list))
    print("SC-SIC:fp:"+str(rNOMA_list)+",SC-SIC:cvx:"+str(cvxNOMA_list))
    print("SC-SIC per group:fp:"+str(rNOMA_group_list)+",SC-SIC per group:cvx:"+str(cvxNOMA_group_list))
    f = open(filename_base_for_save+"result.txt", 'w')
    print("time:GRS:fp:"+str(t_rRSMA_generalized_list)+",GRS:cvx:"+str(t_cvxRSMA_generalized_list), file=f)
    print("time:HRS:fp:"+str(t_rRSMA_twolayers_list)+",HRS:cvx:"+str(t_cvxRSMA_twolayers_list), file=f)
    print("time:1RS:fp:"+str(t_rRSMA_onelayer_list)+",1RS:cvx:"+str(t_cvxRSMA_onelayer_list), file=f)
    print("time:MULP:fp:"+str(t_rMULP_list)+",MULP:cvx:"+str(t_cvxMULP_list), file=f)
    print("time:NOMA:fp:"+str(t_rNOMA_list)+",NOMA:cvx:"+str(t_cvxNOMA_list), file=f)
    print("time:SC-SIC per group:fp:"+str(t_rNOMA_group_list)+",SC-SIC per group:cvx:"+str(t_cvxNOMA_group_list), file=f)

    print("GRS:fp:"+str(rRSMA_generalized_list)+",GRS:cvx:"+str(cvxRSMA_generalized_list), file=f)
    print("HRS:fp:"+str(rRSMA_twolayers_list)+",HRS:cvx:"+str(cvxRSMA_twolayers_list), file=f)
    print("1RS:fp:"+str(rRSMA_onelayer_list)+",1RS:cvx:"+str(cvxRSMA_onelayer_list), file=f)
    print("MULP:fp:"+str(rMULP_list)+",MULP:cvx:"+str(cvxMULP_list), file=f)
    print("SC-SIC:fp:"+str(rNOMA_list)+",SC-SIC:cvx:"+str(cvxNOMA_list), file=f)
    print("SC-SIC per group:fp:"+str(rNOMA_group_list)+",SC-SIC per group:cvx:"+str(cvxNOMA_group_list), file=f)




def atuo_save_info(N_sample,N_channel,tolerance,filename_base,number_of_try,filename_base_for_save,N_users,Nt,SNR,rths,bias,weight,
rho,group,tolerance_inner,maxcount,alpha_imps,
                   alpha_NOMA=[],alpha_NOMA_group=[],alpha_RSMA_onelayer=[],alpha_RSMA_generalized=[],
                   alpha_RSMA_twolayers=[],alpha_MULP=[]):
    contents=''
    filename_base_str='filename_base：'+filename_base+'\n'
    filename_base_for_save_str='filename_base_for_save：'+filename_base_for_save+'\n'
    N_sample_str='N_sample='+str(N_sample)+'\n'
    N_channel_str='N_channel='+str(N_channel)+'\n'
    tolerance_str='tolerance='+str(tolerance)+'\n'
    number_of_try_str='number_of_try='+str(number_of_try)+'\n'
    Nr_str='N_users='+str(N_users)+'\n'
    Nt_str='Nt='+str(Nt)+'\n'
    SNR_str='SNR='+str(SNR)+'\n'
    rths_str='rths='+str(rths)+'\n'
    bias_str='bias='+str(bias)+'\n'
    weight_str='weight='+str(weight)+'\n'
    alpha_imps_str='alpha_imps='+str(alpha_imps)+'\n'
    alpha_NOMA_str='alpha_NOMA='+str(alpha_NOMA)+'\n'
    alpha_NOMA_group_str='alpha_NOMA_group='+str(alpha_NOMA_group)+'\n'
    alpha_RSMA_onelayer_str='alpha_RSMA_onelayer='+str(alpha_RSMA_onelayer)+'\n'
    alpha_RSMA_generalized_str='alpha_RSMA_generalized='+str(alpha_RSMA_generalized)+'\n'
    alpha_RSMA_twolayers_str='alpha_RSMA_twolayers='+str(alpha_RSMA_twolayers)+'\n'
    alpha_MULP_str='alpha_MULP='+str(alpha_MULP)+'\n'
    rho_str='rho='+str(rho)+'\n'
    group_str='group='+str(group)+'\n'
    tolerance_inner_str='tolerance_inner='+str(tolerance_inner)+'\n'
    maxcount='maxcount='+str(maxcount)+'\n'

    contents+=filename_base_str
    contents+=filename_base_for_save_str
    contents+=N_sample_str
    contents+=N_channel_str
    contents+=tolerance_str
    contents+=number_of_try_str
    contents+=Nr_str
    contents+=Nt_str
    contents+=SNR_str
    contents+=rths_str
    contents+=bias_str
    contents+=weight_str
    contents+=alpha_imps_str
    contents+=alpha_MULP_str
    contents+=alpha_NOMA_str
    contents+=alpha_NOMA_group_str
    contents+=alpha_RSMA_onelayer_str
    contents+=alpha_RSMA_twolayers_str
    contents+=alpha_RSMA_generalized_str
    contents+=rho_str
    contents += group_str
    contents += tolerance_inner_str
    contents += maxcount

    f = open(filename_base_for_save+'info.txt','w')
    f.close()

def RSMA_onelayer(H_estimate,H_reals,N_users,tolerance,Pt,Nt,weight,alpha,rho,tolerance_inner,maxcount):
    fpSR,fpT,cvxSR,cvxT=algoritm.RSMA_onelayer(H_estimate,H_reals,N_users,tolerance,rho,Pt,Nt,tolerance_inner,alpha,maxcount,weight)
    return fpSR,fpT,cvxSR,cvxT

def RSMA_twolayers(group,H_estimate,H_reals,N_users,Nt,tolerance,weight,alpha_inner,Pt,rho,alpha,tolerance_inner,maxcount):
    fpSR,fpT,cvxSR,cvxT=algoritm.HRS_full_perm(group, H_estimate, H_reals, N_users, Nt, tolerance, alpha_inner, Pt, rho, alpha, tolerance_inner,
                  maxcount, weight)
    return fpSR,fpT,cvxSR,cvxT

def MULP(H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, weight, alpha,tolerance_inner,maxcount):
    fpSR,fpT,cvxSR,cvxT=algoritm.MULP(H_estimate, H_reals,N_users,Nt,tolerance,rho,Pt,tolerance_inner,alpha,maxcount,weight)
    return fpSR,fpT,cvxSR,cvxT

def NOMA(pro_numb,H_estimate,H_reals,N_users,Nt,tolerance,rho,Pt,weight,alpha,tolerance_inner,maxcount):
    fpSR,fpT,cvxSR,cvxT=algoritm.NOMA_full_perm(pro_numb,H_estimate,H_reals,N_users,Nt,tolerance,rho,Pt,tolerance_inner,maxcount,alpha,weight)
    return fpSR,fpT,cvxSR,cvxT

def NOMA_group(pro_numb,H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, weight, alpha, group,tolerance_inner, maxcount):
    fpSR,fpT,cvxSR,cvxT=algoritm.NOMA_group_full_perm(pro_numb, H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, tolerance_inner, maxcount,
                         alpha, group, weight)
    return fpSR,fpT,cvxSR,cvxT

def RSMA_generalized(pro_numb,H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, weight, alpha, tolerance_inner, maxcount):
    fpSR,fpT,cvxSR,cvxT=algoritm.GRS_full_perm(pro_numb, H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, tolerance_inner, maxcount, alpha,
                  weight)
    return fpSR,fpT,cvxSR,cvxT

def initial_H_random(seed, N_users, Nt,bias):
    H=[]
    np.random.seed(seed)
    for i in range(N_users):
        h=np.random.randn(Nt)+1j * np.random.randn(Nt)
        h=np.sqrt(bias[i])*h*1/np.sqrt(2)
        H.append(np.reshape(h,(Nt,1)))
    H=np.concatenate(H,axis=1)
    return H

def produce_H(seed,N_users,Nt, bias,P_e,Number_of_sample):
    H_estimate = initial_H_random(seed, N_users, Nt, bias) * np.sqrt(1 - P_e)
    H_reals = []
    for seed_i in range(Number_of_sample):
        H_error = initial_H_random(seed_i*25, N_users, Nt, bias) * np.sqrt(P_e)
        H_real = H_estimate + H_error
        H_reals.append(H_real)
    return H_estimate,H_reals

def initial_var_depend_users(N_users):
    if N_users==3:
        bias = [1, 1,1]
        alpha_NOMA = [0.3,0.3,0.4]
        alpha_NOMA_group=[0.3,0.3,0.4]
        alpha_RSMA_generalized = [0.98, 0.01, 0.01]#每一层，共Nr层
        alpha_MULP = [0.3,0.3,0.4]#每个用户，共Nr个
        alpha_RSMA_onelayer = [0.1, 0.9]
        alpha_RSMA_twolayers = [0.1, 0.1, 0.8]#固定3层，公有、组公有、私有
        groups = [2, 1]
    elif N_users==4:
        bias = [1, 1,1,1]
        alpha_NOMA = [0.25,0.25,0.25,0.25]
        alpha_NOMA_group=[0.25,0.25,0.25,0.25]
        alpha_RSMA_generalized = [0.78, 0.2,0.01, 0.01]#每一层，共Nr层
        alpha_MULP = [0.25,0.25,0.25,0.25]#每个用户，共Nr个
        alpha_RSMA_onelayer = [0.1, 0.9]
        alpha_RSMA_twolayers = [0.1, 0.1, 0.8]#固定3层，公有、组公有、私有
        groups = [2, 2]

    return groups,bias,alpha_NOMA,alpha_NOMA_group,alpha_RSMA_generalized,alpha_MULP,alpha_RSMA_onelayer,alpha_RSMA_twolayers

def long_append(i,fp_SR_list,fp_t_list,cvx_SR_list,cvx_t_list,fp_SR,fp_t,cvx_SR,cvx_t):
    fp_SR_list[i].append(fp_SR)
    fp_t_list[i].append(fp_t)
    cvx_SR_list[i].append(cvx_SR)
    cvx_t_list[i].append(cvx_t)
    pass
def create_folder(filename_base):
    try:
        File_Path = os.getcwd() + "\\"+ "data"+"\\" +filename_base +"\\"
        print(File_Path)
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)
            print("目录新建成功：" + File_Path)
        else:
            print("目录已存在！！！")
    except BaseException as msg:
        print("新建目录失败：" + msg)
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pro_numb=8
    alpha_imp=0.6
    Number_of_sample=10
    Nt = 4
    N_users = 3
    rho = 0.5
    tolerance = 1e-3
    weight=[1 for i in range(N_users)]
    tolerance_inner = 10 ** (- 5)
    maxcount = 500
    N_channel=100
    SNRs=[20]
    Pts = [10 ** (SNRs[i] / 10) for i in range(len(SNRs))]
    group,bias, \
    alpha_NOMA, alpha_NOMA_group, alpha_RSMA_generalized, \
    alpha_MULP, alpha_RSMA_onelayer, alpha_RSMA_twolayers=initial_var_depend_users(N_users)
    number_of_try=1
    filename_base='demo--'+str(N_users)+'users_'+str(Nt)+'Nt_'+str(N_channel)+'_tolerance'+str(tolerance)+"_M"+str(Number_of_sample)+"_alpha_imp"+str(alpha_imp)
    create_folder(filename_base)
    filename_base_for_save = "data"+'\\' +filename_base + "\\"+ str(number_of_try) + filename_base
    compare_term_length = len(SNRs)
    fpNOMA_sum_list = [[] for i in range(compare_term_length)]
    fpNOMA_group_sum_list = [[] for i in range(compare_term_length)]
    fpMULP_list_sum = [[] for i in range(compare_term_length)]
    fpRS1_sum_list = [[] for i in range(compare_term_length)]
    fpRSMA_generalized_list_sum = [[] for i in range(compare_term_length)]
    fpRSMA_twolayers_list_sum = [[] for i in range(compare_term_length)]
    fpNOMA_t_list = [[] for i in range(compare_term_length)]
    fpNOMA_group_t_list = [[] for i in range(compare_term_length)]
    fpMULP_list_t = [[] for i in range(compare_term_length)]
    fpRSMA_CMD_list_t = [[] for i in range(compare_term_length)]
    fpRS1_t_list = [[] for i in range(compare_term_length)]
    fpRSMA_generalized_list_t = [[] for i in range(compare_term_length)]
    fpRSMA_twolayers_list_t = [[] for i in range(compare_term_length)]
    cvxNOMA_sum_list = [[] for i in range(compare_term_length)]
    cvxNOMA_group_sum_list = [[] for i in range(compare_term_length)]
    cvxMULP_list_sum = [[] for i in range(compare_term_length)]
    cvxRS1_sum_list = [[] for i in range(compare_term_length)]
    cvxRSMA_generalized_list_sum = [[] for i in range(compare_term_length)]
    cvxRSMA_twolayers_list_sum = [[] for i in range(compare_term_length)]
    cvxNOMA_t_list = [[] for i in range(compare_term_length)]
    cvxNOMA_group_t_list = [[] for i in range(compare_term_length)]
    cvxMULP_list_t = [[] for i in range(compare_term_length)]
    cvxRS1_t_list = [[] for i in range(compare_term_length)]
    cvxRSMA_generalized_list_t = [[] for i in range(compare_term_length)]
    cvxRSMA_twolayers_list_t = [[] for i in range(compare_term_length)]

    atuo_save_info(Number_of_sample,N_channel,tolerance,filename_base,number_of_try,filename_base_for_save,N_users,Nt,SNRs,0,bias,[],
                    rho,group,tolerance_inner,maxcount,alpha_imp,
                    alpha_RSMA_generalized=alpha_RSMA_generalized,
                    alpha_NOMA=alpha_NOMA,alpha_MULP=alpha_MULP,
                    alpha_NOMA_group=alpha_NOMA_group,alpha_RSMA_onelayer=alpha_RSMA_onelayer,
                    alpha_RSMA_twolayers=alpha_RSMA_twolayers)
    for seed in range(0,N_channel):
        print("seed",seed)
        for i in range(compare_term_length):
            print("SNR",SNRs[i])
            Pt=Pts[i]
            P_e = Pts[i] ** (-alpha_imp)
            H_estimate,H_reals=produce_H(seed, N_users, Nt, bias, P_e, Number_of_sample)
            eSR_MULP, et_MULP, cvxSR_MULP, cvxt_MULP=MULP(H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, weight, alpha_MULP, tolerance_inner, maxcount)
            print("MULP:fpSR:",eSR_MULP,",fp_T:",et_MULP,",cvx_SR:",cvxSR_MULP,",cvx_T:",cvxt_MULP)
            eSR_GRS, et_GRS, cvxSR_GRS, cvxt_GRS=RSMA_generalized(pro_numb,H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, weight, alpha_RSMA_generalized, tolerance_inner,maxcount)
            print("GRS:fpSR:", eSR_GRS, ",fp_T:", et_GRS, ",cvx_SR:", cvxSR_GRS, ",cvx_T:", cvxt_GRS)
            eSR_HRS, et_HRS, cvxSR_HRS, cvxt_HRS=RSMA_twolayers(group, H_estimate, H_reals, N_users, Nt, tolerance, weight, [0.25,0.25], Pt, rho, alpha_RSMA_twolayers,tolerance_inner, maxcount)
            print("HRS:fpSR:",eSR_HRS,",fp_T:",et_HRS,",cvx_SR:",cvxSR_HRS,",cvx_T:",cvxt_HRS)
            eSR_NOMA_g, et_NOMA_g, cvxSR_NOMA_g, cvxt_NOMA_g=NOMA_group(pro_numb,H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, weight, alpha_NOMA_group, group, tolerance_inner,maxcount)
            print("NOMA_group:fpSR:",eSR_NOMA_g,",fp_T:",et_NOMA_g,",cvx_SR:",cvxSR_NOMA_g,",cvx_T:",cvxt_NOMA_g)
            eSR_NOMA, et_NOMA, cvxSR_NOMA, cvxt_NOMA=NOMA(pro_numb,H_estimate, H_reals, N_users, Nt, tolerance, rho, Pt, weight, alpha_NOMA, tolerance_inner, maxcount)
            print("NOMA:fpSR:",eSR_NOMA,",fp_T:",et_NOMA,",cvx_SR:",cvxSR_NOMA,",cvx_T:",cvxt_NOMA)
            eSR_1RS, et_1RS, cvxSR_1RS, cvxt_1RS=RSMA_onelayer(H_estimate, H_reals, N_users, tolerance, Pt, Nt, weight, alpha_RSMA_onelayer, rho, tolerance_inner,maxcount)
            print("1-layerRS:fpSR:",eSR_1RS,",fp_T:",et_1RS,",cvx_SR:",cvxSR_1RS,",cvx_T:",cvxt_1RS)
            long_append(i,fpRS1_sum_list,fpRS1_t_list,cvxRS1_sum_list,cvxRS1_t_list,eSR_1RS, et_1RS, cvxSR_1RS, cvxt_1RS)
            long_append(i,fpRSMA_twolayers_list_sum,fpRSMA_twolayers_list_t,cvxRSMA_twolayers_list_sum,cvxRSMA_twolayers_list_t,eSR_HRS, et_HRS, cvxSR_HRS, cvxt_HRS)
            long_append(i,fpMULP_list_sum,fpMULP_list_t,cvxMULP_list_sum,cvxMULP_list_t,eSR_MULP,et_MULP,cvxSR_MULP,cvxt_MULP)
            long_append(i,fpNOMA_sum_list,fpNOMA_t_list,cvxNOMA_sum_list,cvxNOMA_t_list,eSR_NOMA,et_NOMA,cvxSR_NOMA,cvxt_NOMA)
            long_append(i,fpNOMA_group_sum_list,fpNOMA_group_t_list,cvxNOMA_group_sum_list,cvxNOMA_group_t_list,eSR_NOMA_g,et_NOMA_g,cvxSR_NOMA_g,cvxt_NOMA_g)
            long_append(i,fpRSMA_generalized_list_sum,fpRSMA_generalized_list_t,cvxRSMA_generalized_list_sum,cvxRSMA_generalized_list_t,eSR_GRS,et_GRS,cvxSR_GRS,cvxt_GRS)
        store(filename_base_for_save)
        if (seed+1) %5 ==0:
            store_avg(filename_base_for_save)
    pass

