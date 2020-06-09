#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:08:16 2020

@author: Yue He
"""


import pandas as pd
import numpy as np
from scipy.stats import ortho_group
##1.超参数
##产生100个回归数据集1000*40
n_dataset=100
n_sample=100
n_feature=40
n_true=10

def data_generator(n_feature,n_sample):
    fdata=np.zeros((n_sample,n_feature))
    coln=[]
    for i in range(n_feature):
        coln.append('x_'+str(i+1))
        fdata[:,i]=np.random.normal(0,1,n_sample)
    fdata=pd.DataFrame(fdata,columns=coln)
    return fdata


        
##2.定义生成目标函数的函数
#2.1产生从1到n的随机置换
def RandInt(i,j):
	if i==0:
		return np.random.randint(0,10000)%(j+1)
	else:
		return np.random.randint(0,10000)%(j-i+1)+i
 
def zh(a,n):
	for i in range(0,n):
		a[i] = i+1
	# for i in range(0,n):
	# 	print(a[i])
	for i in range(1,n):
		a1 = i
		a2 = RandInt(0,i)
		#print(a1,a2)
		#swap(a1,a2)
		tmp = a[a1]
		a[a1] = a[a2]
		a[a2] = tmp
	 	#print(a1,a2)
	#print(a)
	return a
#2.2 模拟随机协方差矩阵
def cov_matrix(n_l):
    ##创建一个随机正交矩阵
        
    a,b= np.float32(ortho_group.rvs(size=2,dim=n_l, random_state=1))
    ##创建一个随机对角矩阵
    u_a=0.1
    u_b=2
    sqroot_dl=np.random.uniform(u_a,u_b,n_l)
    d_l=sqroot_dl**2
    D_l=np.diag(d_l)
    #三个矩阵相乘
    return np.dot(np.dot(a,D_l),a.T)

#2.3计算样本的每个小函数的值
def h_l(n_l,x_l):
    mu_l=np.random.normal(0,1,n_l)
    a=x_l-mu_l
    V_inverse=np.linalg.inv(cov_matrix(n_l))
    data_sum=np.dot(np.dot(a,V_inverse),a.T)
    hfunc=np.exp(-0.5*data_sum)
    return hfunc

##3.对数据集计算目标函数值
def finaldata(n_feature,n_sample,n_true):
    sim_data=data_generator(n_feature,n_sample)
    coefs=np.random.uniform(-1,1,n_true) 
    F=[]
    for i in range(n_sample):
        w=sim_data.iloc[i,:]
        hlist=[]
        for j in range(n_true):
            r=np.random.exponential(2)
            n_l=int(np.floor(2.5+r))
            A=np.zeros(n_true)
            P_l=zh(A,n_true)
            index=np.array(P_l-1).astype('int32')
            index=np.random.choice(index,n_l)
            x_l=w[index]
            e=h_l(n_l,x_l)
            hlist.append(e)
        target=np.dot(coefs,hlist)    
        print(target)
        F.append(target)
    sigma=np.std(F)
    error=np.random.normal(0,sigma,n_sample)
    true_value=F+error
    sim_data['F_value']=true_value
    return sim_data

def dataset_generator(n_dataset,n_feature,n_sample,n_true):
    dataset={}
    for i in range(n_dataset):
        dataname='fdata'+str(i+1)
        dataset[dataname]=finaldata(n_feature,n_sample,n_true)
    return dataset
 
## 4.开始使用函数
#4.1生成一个数据集
d1=finaldata(n_feature,n_sample,n_true)

##生成100个这样的数据集
dataset_generator(n_dataset,n_feature,n_sample,n_true)
