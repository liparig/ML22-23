import matplotlib.pyplot as plt
from matplotlib import rc
import datetime

from costants import FORMATTIMESTAMP, LABEL_PLOT_TRAINING, LABEL_PLOT_VALIDATION

# importing the module
from memory_profiler import profile
    
"""from matplotlib import figure
fig = figure.Figure()
ax = fig.subplots(1)
Also no need to do plt.close() or anything. It worked for me."""
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
                    caption+= r"$\tau:\,"+value[0]+r"\,\eta_\tau:\,"+str(value[1])+"$, "
            case 'reg':
                if value[0]!=False:
                    caption+= "$Reg \,"+value[0].title()+r"\,\lambda=\,"+str(value[1])+"$, "
            case "dim_batch":
                caption+= r"$Batch size:\,"+str(value)+"$ "
            case'momentum':
                if value[0]!=False:
                    caption+= "$Momentum:\, "+value[0].title()+r"\,\alpha: "+str(value[1])+"$, "
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
            case _:
                pass
        
    caption= r"\begin{center} \textit{\small{" + caption + r"}} \end{center}"
    return caption
@profile
def plot_curves(loss_tr, loss_vs, metr_tr, val_metr, path = None, ylim = (0., 10.), lbl_tr:str = LABEL_PLOT_TRAINING,
                lbl_vs:str = LABEL_PLOT_VALIDATION, titleplot:str = '',
                theta:dict ={}, labelsX:list[str] = ['Epochs','Epochs'], labelsY:list[str] = ['Loss','Metric']):
    """
    Plot the curves of training loss, training metric, validation loss, validation metric
    :param loss_tr: vector with the training error values
    :param loss_vs: vector with the validation error values
    :param metr_tr: vector with the training metric values
    :param val_metr: vector with the validation metric values
    :param path: if not None, path where to save the plot (otherwise it will be displayed)
    :param ylim: value for "set_ylim" of pyplot
    :param lbl_tr: label for the training curve
    :param lbl_vs: label for the validation curve
    """
    plt.close('all')
    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    
    caption=__theta_toCaption(theta)

    figure, ax = plt.subplots(1, 2, figsize = (18, 6))
    
    ax[0].grid(True)
    ax[1].grid(True)
    """
    if(len(titlesSubplot) > 1):
        ax[0].set_title(titlesSubplot[0])
        ax[1].set_title(titlesSubplot[1])"""
   
    ax[0].plot(range(len(loss_tr)), loss_tr, color='b', linestyle='dashed', label=lbl_tr)
    ax[0].plot(range(len(loss_vs)), loss_vs, color='r', label=lbl_vs)
    ax[0].legend(loc='best', prop={'size': 9})
    ax[0].set_xlabel(labelsX[0], fontweight='bold')
    #ax[0].set_xlabel(labelsX[0] + "\\*" + caption ,fontweight='bold')
    ax[0].set_ylabel(labelsY[0], fontweight='bold')
    
    ax[1].plot(range(len(metr_tr)), metr_tr, color='b', linestyle='dashed', label=lbl_tr)
    ax[1].plot(range(len(val_metr)), val_metr, color='r', label=lbl_vs)
    ax[1].legend(loc='best', prop={'size': 9})
    ax[1].set_xlabel(labelsX[1], fontweight='bold')
    ax[1].set_ylabel(labelsY[1], fontweight='bold')
    ax[1].set_ylim(ylim)

    plt.suptitle(titleplot+"\n"+caption)
    plt.subplots_adjust(hspace=0.5)
    
    if path is None:
        plt.show()
    else:
        s = f'{path}{datetime.datetime.now().strftime(FORMATTIMESTAMP)}.jpg'
        plt.savefig(s)
        
    plt.close()
    figure.clf()