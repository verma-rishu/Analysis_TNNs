import numpy as np
import matplotlib.pyplot as plt


def plot_models_vs_lr(mean_error_evolvegcno,mean_error_evolvegcnh,mean_error_gconvgru,mean_error_gconvlstm):
    lr = [0.1, 0.01, 0.001, 0.0001]
    marker = ['o','>','x','o','<']
    list_of_MSE = [mean_error_evolvegcno,mean_error_evolvegcnh,mean_error_gconvgru,mean_error_gconvlstm]
    models_name = ['evolvegcno','evolvegcnh','gconvgru','gconvlstm']

    for list_idx, model_MSE_list in enumerate(list_of_MSE):
        for lr_idx, lr_val in enumerate(lr):
            x_values = range(len(model_MSE_list[lr_idx]))
            label = "lr = "+str(lr_val)
            plt.plot(x_values, model_MSE_list[lr_idx], label=label, marker=marker[lr_idx])
            
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Mean Square Error')
        plt.title(f'Performance of {models_name[list_idx]} with varying lr')
        plt.savefig('plots/'+models_name[list_idx]+'_lr.png',  dpi=1000)
        plt.cla()
