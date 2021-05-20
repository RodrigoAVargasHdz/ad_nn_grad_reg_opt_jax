import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as f_mse
from data import *
Ha2cm = 219474.6#220000
Ha2kcal = 627.5
Ha2eV = 27.211

def f_min(N,str_nn_arq,r_dir):
    
    loss0 = 1E6
    D0 = []
    f0 = []
    files_all = os.listdir(r_dir)
    for f in files_all:
        if 'data' in f and str(N) in f:
            Dt = np.load(r_dir + '/' + f,allow_pickle=True)
            loss = Dt.item()['error_full']
            if loss < loss0:
                D0 = Dt
                f0 = f
                loss0 = loss
    return D0, f0

def f_plot_exact_data():
    Dt = Data_nh3()
    Xt,gXt,gXct,yt = Dt
    
    x = np.arange(yt.shape[0])
    fig, axs = plt.subplots(2,sharex=True)
#     fig.suptitle('Vertically stacked subplots')
    axs[0].set_title('Ground state (ad)')
    axs[0].scatter(x, yt[:,0])

    axs[1].set_title('Excited state (ad)')
    axs[1].scatter(x, yt[:,1])
    plt.savefig('fig_temp.png')
    
# ENERGY
def f_plot_y_pred_vs_exact(file):
    D = jnp.load(file,allow_pickle=True)
    Dpred = D.item()['Dpred'] 
    X,y_pred = Dpred[:,:-2],Dpred[:,-2:]
    
    Dt = Data_nh3()
    Xt,gXt,gXct,yt = Dt
    
    diff_y = jnp.abs(y_pred - yt)
    x = np.arange(diff_y.shape[0])
    
    fig, axs = plt.subplots(2,sharex=True)
    axs[0].set_title('Ground state (ad)')
#     axs[0].scatter(x, diff_y[:,0])
    axs[0].scatter(x, y_pred[:,0])

    axs[1].set_title('Excited state (ad)')
#     axs[1].scatter(x, diff_y[:,1])
    axs[1].scatter(x, y_pred[:,1])


    plt.savefig('fig_y_pred.png')
#     plt.show()

def f_plot_y_pred_vs_exact_mae(file):
    D = jnp.load(file,allow_pickle=True)
    Dpred = D.item()['Dpred'] 
    X,y_pred = Dpred[:,:-2],Dpred[:,-2:]
    
    Dt = Data_nh3()
    Xt,gXt,gXct,yt = Dt
    
    diff_y = jnp.abs(y_pred - yt)
    x = np.arange(diff_y.shape[0])
    mse0 =  np.sqrt(f_mse(y_pred[:,0], yt[:,0]))
    mse1 =  np.sqrt(f_mse(y_pred[:,1], yt[:,1]))
    print(mse0,mse1)

    ind = np.argsort(diff_y, axis=0)
    Z = np.column_stack((diff_y[ind[:,0],0],diff_y[ind[:,1],1]))
    
    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(2,sharex=True)
    axs[0].set_title('Ground state (ad)')
    axs[0].semilogy(x, diff_y[:,0],ls='None',marker='o',markersize=3)
    axs[0].set_ylabel('MAE')
#     sc0 = axs[0].scatter(x, y_pred[:,0],c=diff_y[:,0],vmin=np.amin(diff_y[:,0]), vmax=np.amax(diff_y[:,0]), s=5, cmap=cm)
#     cbar0 = fig.colorbar(sc0,ax=axs[0])  
#     cbar0.set_label("MAE")
#     axs[0].set_ylabel('Energy')
    
    axs[1].set_title('Excited state (ad)')
    axs[1].semilogy(x, diff_y[:,1],ls='None',marker='o',markersize=3)
    axs[1].set_ylabel('MAE')
    axs[1].set_xlabel('Data')
    
#     sc1 = axs[1].scatter(x, y_pred[:,1],c=diff_y[:,1],vmin=np.amin(diff_y[:,1]), vmax=np.amax(diff_y[:,1]), s=5, cmap=cm)
#     cbar1 = fig.colorbar(sc1,ax=axs[1]) 
#     axs[1].set_ylabel('Energy')
#     axs[1].set_xlabel('Data')
#     cbar1.set_label("MAE") 
    
    plt.tight_layout()
    plt.savefig('fig_scatter_y_pred.png')   
#     plt.show()

def plot_y_pred_histogram(file):
    D = jnp.load(file,allow_pickle=True)
    Dpred = D.item()['Dpred'] 
    X,y_pred = Dpred[:,:-2],Dpred[:,-2:]
    
    Dt = Data_nh3()
    Xt,gXt,gXct,yt = Dt
    
    diff_y = jnp.abs(y_pred - yt)
    x = np.arange(diff_y.shape[0])    

    n_bins = 50
    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
    axs[0].set_title('Ground state (ad)')
    hist0, bins0 = np.histogram(diff_y[:,0], bins=n_bins)
    logbins0 = np.logspace(np.log10(bins0[0]),np.log10(bins0[-1]),len(bins0))
    axs[0].hist(diff_y[:,0], bins=logbins0)
#     axs[0].set_xticks([1e-6,1E-5,1E-4,1E-3,1E-2,1E-1])
#     axs[0].set_xticklabels([r'$10^{-%i}$'%(i) for i in range(-6,0)])
    axs[0].set_xlim([1E-7, 2E-1])
    axs[0].set_xscale('log')
    axs[0].set_xlabel('MAE')
    
    axs[1].set_title('Excited state (ad)')
    hist1, bins1 = np.histogram(diff_y[:,1], bins=n_bins)
    logbins1 = np.logspace(np.log10(bins1[0]),np.log10(bins1[-1]),len(bins1))
    axs[1].hist(diff_y[:,1], bins=logbins1)
#     axs[1].set_xticks([1e-6,1E-5,1E-4,1E-3,1E-2,1E-1])
#     axs[1].set_xticklabels([r'$10^{-%i}$'%(i) for i in range(-6,0)])
    axs[1].set_xlim([1E-7, 2E-1])
    axs[1].set_xscale('log')
    axs[1].set_xlabel('MAE')

    plt.savefig('fig_histogram_y_pred.png')   
#     plt.show()

def plot_y_pred_histogram_multiple():

    act_fun = 'reLU'
    r_dir = 'Results_{}/Results_nn_adiab'.format(act_fun)
    N = 4300
    str_nn_arq = '550_550_100'#'500_250_100'
    D0,f0 = f_min(N,str_nn_arq,r_dir)
    print(f0)
    f0 = r_dir + '/' + f0
    
    r_dir = 'Results_{}/Results_nn_coulomb_adiab'.format(act_fun)
    N = 4300
    str_nn_arq = '550_550_100'#'500_250_100'
    D1,f1 = f_min(N,str_nn_arq,r_dir)
    print(f1)
    f1 = r_dir + '/' + f1

    r_dir = 'Results_{}/Results_nn_pip_adiab'.format(act_fun)   #'Results_nn_morse_adiab'
    N = 4300
    str_nn_arq = '550_550_100'#'500_250_100'
    D2,f2 = f_min(N,str_nn_arq,r_dir)
    print(f2)
    f2 = r_dir + '/' + f2
        
    r_dir = 'Results_{}/Results_nn_morse_adiab'.format(act_fun)   #'Results_nn_morse_adiab'
    N = 4300
    str_nn_arq = '550_550_100'#'500_250_100'
    D3,f3 = f_min(N,str_nn_arq,r_dir)
    print(f2)
    f3 = r_dir + '/' + f3
        


    def histogram_bins(y_pred):   
        Dt = Data_nh3()
        Xt,gXt,gXct,yt = Dt
    
        diff_y = jnp.abs(Ha2eV*(y_pred - yt))
        x = np.arange(diff_y.shape[0]) 
        n_bins = 50  
        hist0, bins0 = np.histogram(diff_y[:,0], bins=n_bins) 
        logbins0 = np.logspace(np.log10(bins0[0]),np.log10(bins0[-1]),len(bins0))
 
        hist1, bins1 = np.histogram(diff_y[:,1], bins=n_bins)
        logbins1 = np.logspace(np.log10(bins1[0]),np.log10(bins1[-1]),len(bins1))
        return logbins0, logbins1, diff_y

    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
    axs[0].set_title('Ground state')
    N_ = ['R','Coulomb Matrix','PIP',r'PIP$_{2}$']
    for i,f in enumerate([f0,f1,f2,f3]):
        D = jnp.load(f,allow_pickle=True)
        Dpred = D.item()['Dpred'] 
        X,y_pred = Dpred[:,:-2],Dpred[:,-2:]
        logbins0, logbins1, diff_y = histogram_bins(y_pred)
        axs[0].hist(diff_y[:,0], bins=logbins0,alpha=0.5,label='%s'%N_[i])
        axs[1].hist(diff_y[:,1], bins=logbins1,alpha=0.5,label='%s'%N_[i])
        
    axs[0].set_xlim([1E-5, 1.])
    axs[0].set_xscale('log')
    axs[0].set_xlabel('MAE [eV]',fontsize=15)
    axs[0].legend()
    
    axs[1].set_title('Excited state')
    axs[1].set_xlim([1E-5, 1.])
    axs[1].set_xscale('log')
    axs[1].set_xlabel('MAE [eV]',fontsize=15)
#     axs[1].legend()
    plt.tight_layout()
    plt.savefig('fig_histogram_y_pred_{}.png'.format(act_fun)) 
# -------------------------------------  

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Neural Network')
    
def plot_y_pred_violin_multiple():
    def f_best_results(act_fun):
        r_dir = 'Results_{}/Results_nn_adiab'.format(act_fun)
        N = 4300
        str_nn_arq = '550_550_100'#'500_250_100'
        D0,f0 = f_min(N,str_nn_arq,r_dir)
        print(f0)
        f0 = r_dir + '/' + f0
    
        r_dir = 'Results_{}/Results_nn_coulomb_adiab'.format(act_fun)
        N = 4300
        str_nn_arq = '550_550_100'#'500_250_100'
        D1,f1 = f_min(N,str_nn_arq,r_dir)
        print(f1)
        f1 = r_dir + '/' + f1

        r_dir = 'Results_{}/Results_nn_pip_adiab'.format(act_fun)
        N = 4300
        str_nn_arq = '550_550_100'#'500_250_100'
        D2,f2 = f_min(N,str_nn_arq,r_dir)
        print(f2)
        f2 = r_dir + '/' + f2

        r_dir = 'Results_{}/Results_nn_morse_adiab'.format(act_fun)
        N = 4300
        str_nn_arq = '550_550_100'#'500_250_100'
        D3,f3 = f_min(N,str_nn_arq,r_dir)
        print(f3)
        f3 = r_dir + '/' + f3
        return [f0,f1,f2,f3]
    
    def f_violins(f,i): 
        D = jnp.load(f,allow_pickle=True)
        Dt = Data_nh3()
        Xt,gXt,gXct,yt = Dt

        Dpred = D.item()['Dpred'] 
        X,y_pred = Dpred[:,:-2],Dpred[:,-2:]
    
        diff_y = jnp.abs(Ha2eV*(y_pred - yt))
        return diff_y[:,i]

    f_all_siLU = f_best_results('reLU')
    f_all_tanh = f_best_results('tanh')   

    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(1,2, tight_layout=True)
    axs[0].set_title('Ground state')
    NN_ = ['R','CM','PIP',r'PIP$_{2}$']
    
    data0_siLU = [f_violins(fi,0) for i,fi in enumerate(f_all_siLU)]
    data1_siLU = [f_violins(fi,1) for i,fi in enumerate(f_all_siLU)]

    data0_tanh = [f_violins(fi,0) for i,fi in enumerate(f_all_tanh)]
    data1_tanh = [f_violins(fi,1) for i,fi in enumerate(f_all_tanh)]

    
    axs[0].violinplot(data0_tanh,showmeans=True,showextrema=False)#,label='tanh'
    axs[0].scatter(0.,np.mean(data0_tanh),label='tanh')
    axs[1].violinplot(data1_tanh,showmeans=True,showextrema=False)#,label='tanh'
    axs[1].scatter(0.,np.mean(data1_tanh),label='tanh')


    axs[0].violinplot(data0_siLU,showmeans=True,showextrema=False)#label='siLU'
    axs[0].scatter(0.,np.mean(data0_siLU),label='reLU')
    axs[1].violinplot(data1_siLU,showmeans=True,showextrema=False)#,label='siLU'
    axs[1].scatter(0.,np.mean(data1_siLU),label='reLU')


    for ax in [axs[0],axs[1]]:
        set_axis_style(ax,NN_)
        
    axs[0].set_ylim([-.025, 0.4])
    axs[0].set_ylabel('MAE [eV]',fontsize=15)
    axs[0].legend()
    
    axs[1].set_title('Excited state')
    axs[1].set_ylim([-.025, 0.4])
#     axs[1].set_ylabel('MAE [eV]',fontsize=15)
#     axs[1].legend()
    plt.tight_layout()
    plt.savefig('fig_violin_y_pred.png') 
# -------------------------------------  

# GRAD
def plot_gX_pred_histogram():
    r_dir = 'Results_nn_adiab/Results_rhoG_0.0005'
    N = 4000
    str_nn_arq = '1000_500_100'#'500_250_100'
    D0,f0 = f_min(N,str_nn_arq,r_dir)
    print(f0)
    f0 = r_dir + '/' + f0
    
    r_dir = 'Results_nn_adiab'
    N = 4000
    str_nn_arq = '1000_500_100'#'500_250_100'
    D1,f1 = f_min(N,str_nn_arq,r_dir)
    print(f1)
    f1 = r_dir + '/' + f1
        
    def histogram_bins(gX_pred):   
        Dt = Data_nh3()
        Xt,gXt,gXct,yt = Dt
        print(gXt.shape,gX_pred.shape)
    
        diff_gX = gX_pred - gXt
        norm_diff_gX = np.linalg.norm(diff_gX,axis=2,ord=1)
        print(norm_diff_gX.shape)
        
        x = np.arange(norm_diff_gX.shape[0]) 
        n_bins = 50  
        hist0, bins0 = np.histogram(norm_diff_gX[:,0], bins=n_bins) 
        logbins0 = np.logspace(np.log10(bins0[0]),np.log10(bins0[-1]),len(bins0))
 
        hist1, bins1 = np.histogram(norm_diff_gX[:,1], bins=n_bins)
        logbins1 = np.logspace(np.log10(bins1[0]),np.log10(bins1[-1]),len(bins1))
        return logbins0, logbins1, norm_diff_gX

    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
    axs[0].set_title('Ground state (ad)')
    N_ = ['NN','Morse-NN']
    for i,f in enumerate([f0,f1]):
        D = jnp.load(f,allow_pickle=True)
        gXpred = D.item()['gXpred'] 
        logbins0, logbins1, diff_y = histogram_bins(gXpred)
        axs[0].hist(diff_y[:,0], bins=logbins0,alpha=0.5,label='%s'%N_[i])
        axs[1].hist(diff_y[:,1], bins=logbins1,alpha=0.5,label='%s'%N_[i])
        
#     axs[0].set_xlim([1E-7, 2E-1])
    axs[0].set_xscale('log')
    axs[0].set_xlabel('MAE')
    axs[0].legend()
    
    axs[1].set_title('Excited state (ad)')
#     axs[1].set_xlim([1E-7, 2E-1])
    axs[1].set_xscale('log')
    axs[1].set_xlabel('MAE')
    axs[1].legend()

    plt.savefig('fig_histogram_gX_pred.png') 

 
def main_energy_error_analysis():
#     r_dir = 'Results_nn_adiab/Results_rhoG_0.0005'
#     r_dir = 'Results_nn_morse_adiab/Results_rhoG_0.001'
#     r_dir = 'Results_nn_adiab'
    r_dir = 'Results_nn_morse_adiab'
#     r_dir = 'Results_nn_coulomb_adiab'
    

    N = 4300
    str_nn_arq = '250_250_100_100_50'#'500_250_100'#'1000_500_100'
    D,f = f_min(N,str_nn_arq,r_dir)
    print(f)

    D = jnp.load(r_dir +'/' + f,allow_pickle=True)
    Dpred = D.item()['Dpred']     
    y_pred = Dpred[:,-2:]    
    
    Dt = Data_nh3()
    Xt,gXt,gXct,yt = Dt 
    
    diff_y  = y_pred - yt
    print(np.amax(np.abs(diff_y),axis=0))
    for dy,y,ytrue in zip(diff_y,y_pred,yt):
        print(dy,y,ytrue)
    
#     f_plot_y_pred_vs_exact(r_dir +'/' + f)
#     plot_y_pred_histogram(r_dir +'/' + f)

    energy_max = 1E3*np.arange(10,100,10)
    i0 = np.append(np.argmin(yt[:,0]),np.argmin(yt[:,1]))
    yt_min = np.append(yt[i0[0],0],yt[i0[1],1])
    yt_ = yt - yt_min
    yt = Ha2cm*yt
    for emax in energy_max:
        j0 = np.where(yt[:,0] <= emax)[0]
        yt0 = yt[j0,0]
        j1 = np.where(yt[:,1] <= emax)[0]
        yt1 = yt[j1,1]
        print(emax, yt0.shape,np.mean(np.abs(diff_y[j0,0])),yt1.shape,np.mean(np.abs(diff_y[j1,1])))
        
            
def main():
    D0 = np.random.rand(3,2)
    print(D0)
    D1 = np.random.rand(3,2)
    print(D1)
    D = D0 - D1
    print(np.linalg.norm(D,axis=1,ord=1))
    
    


if __name__ == "__main__":
#     main()
#     plot_y_pred_histogram_multiple()
    plot_y_pred_violin_multiple()
#     main_energy_error_analysis()

    