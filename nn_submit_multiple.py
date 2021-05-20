import os
import time
import numpy as np

w_decay_ = [1E-3,1E-4,1E-5,5E-4,1E-6,1E-7]
rho_G_ = [0.]#w_decay_
def sh_file(P,N,w_decay,rho_G,i0,nn_arq,acq_f,l):

#     BE SURE TO NAME FOR EACH CALCULATION AN INDIVIDUAL XXX.SH FILE CAUSE THAT IS THE ONE
#     YOU WILL BE SUBMITTING
    if P == 1:
        f_tail = 'nh3_nn_{}_N_{}_wdecay_{}_rhoG_{}_{}_flax_l_{}'.format(nn_arq,N,w_decay,rho_G,acq_f,l)
    if P == 2:
        f_tail = 'nh3_nn_morse_{}_N_{}_wdecay_{}_rhoG_{}_{}_l_{}_flax'.format(nn_arq,N,w_decay,rho_G,acq_f,l)
    if P == 3:
        f_tail = 'nh3_nn_coulomb_{}_N_{}_wdecay_{}_rhoG_{}_{}_l_{}_flax'.format(nn_arq,N,w_decay,rho_G,acq_f,l)
    if P == 4:
        f_tail = 'nh3_nn_pip_{}_N_{}_wdecay_{}_rhoG_{}_{}_l_{}_flax'.format(nn_arq,N,w_decay,rho_G,acq_f,l)

    	
    f=open('JC_%s.sh'%(f_tail), 'w+')	
    f.write('#!/bin/bash \n')
    f.write('#SBATCH --nodes=1 \n')
    f.write('#SBATCH --ntasks=1 \n')
    f.write('#SBATCH --cpus-per-task=12  # Cores proportional to GPU \n')
    f.write('#SBATCH --mem=32G \n')#64
    f.write('#SBATCH --qos nopreemption  \n')
    f.write('#SBATCH --partition=cpu \n')
    f.write('#SBATCH --job-name={} \n'.format(f_tail))
    f.write('#SBATCH --time=4:00:00 \n')
    f.write('#SBATCH --output=out_{}.log \n'.format(f_tail))

    f.write('\n')
#     LOAD MODULES
    f.write('source $HOME/jaxenv/bin/activate \n')

    f.write('\n')

    if P == 1:
        f.write('python3 nn_grad_nh3_batch_flax.py --N {} --list "{}"  --f {} --lr 2E-4 --wdecay {} --lG {} --n_epoch 25000 --i {} --l {} \n'.format(N,nn_arq,acq_f,w_decay,rho_G,i0,l)) 
#         f.write('python3 nn_grad_nh3_flax.py --N {} --list "{}"  --f {} --lr 2E-4 --wdecay {} --lG {} --n_epoch 150000 --i {} --l {} \n'.format(N,nn_arq,acq_f,w_decay,rho_G,i0,l)) 
    if P == 2:
        f.write('python3 nn_morse_grad_nh3_batch_flax.py --N {} --list "{}"  --f {} --lr 2E-4 --wdecay {} --lG {} --n_epoch 20000 --i {} --l {} \n'.format(N,nn_arq,acq_f,w_decay,rho_G,i0,l)) 
    if P == 3:
        f.write('python3 nn_coulomb_grad_nh3_batch_flax.py --N {} --list "{}"  --f {} --lr 2E-4 --wdecay {} --lG {} --n_epoch 20000 --i {} --l {} \n'.format(N,nn_arq,acq_f,w_decay,rho_G,i0,l)) 
    if P == 4:
        f.write('python3 nn_pip_grad_nh3_batch_flax.py --N {} --list "{}"  --f {} --lr 2E-4 --wdecay {} --lG {} --n_epoch 25000 --i {} --l {} \n'.format(N,nn_arq,acq_f,w_decay,rho_G,i0,l)) 


    f.write('\n')
    f.close()
	
    if os.path.isfile('JC_%s.sh'%(f_tail)):
        print('Submitting JC_%s.sh'%(f_tail))
        os.system('sbatch JC_%s.sh '%(f_tail))        


def main():
    N_ = np.array([4300,4000,3000])#3000,4000,4300
    P = 3 # vanillaNN, morseNN, coulombNN
    j0 = 0 # rhoG     
    acq_fun_ = ['tanh','reLU']

    n_layer_ = [550]#250,350,
    for i0 in range(5): #w_decay_
        for nl in n_layer_:
            for n in N_[:1]:
                n_layer = nl
                for l in range(2):
#                     nn_arq_str = '{},{},{}'.format(n_layer,int(n_layer/2),100)
                    nn_arq_str = '{},{},{}'.format(n_layer,n_layer,100)
#                     nn_arq_str = '{},{},{},{},{}'.format(n_layer,n_layer,100,100,50)
                    sh_file(P,n,w_decay_[i0],rho_G_[j0],i0,nn_arq_str,acq_fun_[1],l)
#                     assert 0


if __name__== "__main__":
    main()	
	                
	                
	    
