import matplotlib.pyplot as plt
import datetime

import numpy as np

import costants as C

import multiprocessing as mp

# Turn the configuration object theta in caption for the images of the chart
# :param: theta is the configuration object
# :return: the caption to add in the plot image
def __theta_toCaption(theta:dict):
    caption=""
    for param, value in theta.items():
        match param:
            case 'l_dim':
                caption +=r"$Units:\,"+str(value)+"$, "
            case 'a_functions':
                caption +=r"$Activation Layer:[ \,"+','.join([str(elem).title() for elem in value])+"\,]$, "
            case 'eta':
                caption +=r"$\eta:"+str(value)+"$, "
            case 'tau':
                if value[0] !=False:
                    caption+= r"$\tau:\,"+str(value[0])+r"\,\eta_\tau:\,"+str(value[1])+"$, "
            case 'reg':
                if value[0] !=False:
                    caption+= "$Reg \,"+str(value[0].title())+r"\,\lambda=\,"+str(value[1])+"$, "
            case "dim_batch":
                caption+= r"$Batch size:\,"+str(value)+"$ "
            case'momentum':
                if value[0] !=False:
                    caption+= "$Momentum:\, "+ value[0].title() +r"\,\alpha: "+str(value[1])+"$, "
            case "epochs":
                caption+= r"$MaxEpochs:\,"+str(value)  +"$, "
            case "eps":
                caption+= r"$\epsilon:\,"+str(value)  +"$, "
            case "distribution":
                caption+=r"$"+str(value)+"$, "
            case "bias":
                caption+=r"$Bias:\,"+str(value)+"$, "
            case "early_stop":
                if value:
                    caption+=r"$EarlyStop:\,"+str(theta["treshold_variance"])+"$, "
            case "patience":
                caption+=r"$Patience:\,"+str(value)+"$ "
        
    caption = r"\begin{center} \textit{\small{" + caption + r"}} \end{center}"
    return caption

# Use a additional process for make the plot. In a linux system is more faster than windows os
# :param: loss_tr is vector with the training error values
# :param: loss_vs is vector with the validation error values
# :param: metr_tr is vector with the training metric values
# :param: val_metr is vector with the validation metric values
# :param: error_tr is a flag for print the loss values. Default = false
# :param: path if not None, path is the location where to save the plot (otherwise it will be displayed)
# :param: ylim is the value for "set_ylim" of pyplot in accuracy and metrics plot
# :param: yMSElim is the value for "set_ylim" of pyplot in training/validation/test plot
# :param: lbl_tr is the label for the training curve
# :param: lbl_vs is label for the validation curve
# :param: titlePlot is the title for the plot
# :param: theta is the configuration object
# :param: labelsX is the list of the label on the x axes
# :param: labelsY is the list of the label on the y axes
def plot_curves(loss_tr, loss_vs, metr_tr, val_metr, error_tr = False, path = None, ylim = (0., 10.), yMSElim = (0, 0), lbl_tr:str = C.LABEL_PLOT_TRAINING,
                lbl_vs:str = C.LABEL_PLOT_VALIDATION, titlePlot:str = '',
                theta:dict = {}, labelsX:list[str] = ['Epochs','Epochs'], labelsY:list[str] = ['Loss','Metric']):
    proc=mp.Process(target=draw, args=(loss_tr, loss_vs, metr_tr, val_metr,error_tr, path, ylim ,yMSElim, lbl_tr,lbl_vs, titlePlot, theta, labelsX, labelsY))
    proc.daemon=False
    proc.start()
    proc.join()
    

# Plot the curves of training loss, training metric, validation loss, validation metric
# :param: loss_tr is vector with the training error values
# :param: loss_vs is vector with the validation error values
# :param: metr_tr is vector with the training metric values
# :param: val_metr is vector with the validation metric values
# :param: error_tr is a flag for print the loss values. Default = false
# :param: path if not None, path is the location where to save the plot (otherwise it will be displayed)
# :param: ylim is the value for "set_ylim" of pyplot in accuracy and metrics plot
# :param: yMSElim is the value for "set_ylim" of pyplot in training/validation/test plot
# :param: lbl_tr is the label for the training curve
# :param: lbl_vs is label for the validation curve
# :param: titlePlot is the title for the plot
# :param: theta is the configuration object
# :param: labelsX is the list of the label on the x axes
# :param: labelsY is the list of the label on the y axes
def draw(loss_tr, loss_vs, metr_tr, val_metr, error_tr = False, path = None, ylim = (0., 10.), yMSElim = (0, 0), lbl_tr:str = C.LABEL_PLOT_TRAINING,
                lbl_vs:str = C.LABEL_PLOT_VALIDATION, titlePlot:str = '',
                theta:dict = {}, labelsX:list[str] = ['Epochs','Epochs'], labelsY:list[str] = ['Loss','Metric']):
    
    plt.close('all')
    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    
    caption = __theta_toCaption(theta)

    _, ax = plt.subplots(1, 2, figsize = (18, 6))
    ax[0].grid(True)
    ax[1].grid(True)

   
    ax[0].plot(range(len(loss_tr)), loss_tr, color='b', linestyle = 'dashed', label = lbl_tr)
    ax[0].plot(range(len(loss_vs)), loss_vs, color='r', label = lbl_vs)
    if error_tr:
        ax[0].plot(range(len(error_tr)), error_tr, color='y', label="Loss TR")

    ax[0].legend(loc='best', prop={'size': 9})
    ax[0].set_xlabel(labelsX[0], fontweight='bold')
    #ax[0].set_xlabel(labelsX[0] + "\\*" + caption ,fontweight='bold')
    ax[0].set_ylabel(labelsY[0], fontweight='bold')
    if yMSElim!= (0,0) and (not np.isnan(yMSElim[1])):
        ax[0].set_ylim(yMSElim)
    
    ax[1].plot(range(len(metr_tr)), metr_tr, color='b', linestyle='dashed', label=lbl_tr)
    ax[1].plot(range(len(val_metr)), val_metr, color='r', label=lbl_vs)
    ax[1].legend(loc='best', prop={'size': 9})
    ax[1].set_xlabel(labelsX[1], fontweight='bold')
    ax[1].set_ylabel(labelsY[1], fontweight='bold')
    ax[1].set_ylim(ylim)

    plt.suptitle(f'{titlePlot}\n{caption}')
    plt.subplots_adjust(hspace=0.5)
    
    if path is None:
        plt.show()
    else:
        s = f'{path}{datetime.datetime.now().strftime(C.FORMATTIMESTAMP)}.jpg'
        plt.savefig(s)
    
    plt.clf()
    plt.close()