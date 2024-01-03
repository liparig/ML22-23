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
            case C.L_NET:
                caption += r"$Units:\," + str(value) + "$, "
            case C.L_ACTIVATION:
                caption += (
                    r"$Activation Layer:[ \,"
                    + ','.join([str(elem).title() for elem in value])
                    + "\,]$, "
                )
            case C.L_ETA:
                caption += r"$\eta:" + str(value) + "$, "
            case C.L_TAU:
                if value[0] != False:
                    caption += (
                        r"$\tau:\,"
                        + str(value[0])
                        + r"\,\eta_\tau:\,"
                        + str(value[1])
                        + "$, "
                    )
            case C.L_G_CLIPPING:
                if value[0] != False:
                    caption += r"$Clipping:\," + str(value[1]) + "$, "
            case C.L_DROPOUT:
                if value[0] != False:
                    caption += r"$Dropout p:\," + str(value[1]) + "$, "
            case C.L_REG:
                if value[0] != False:
                    caption += (
                        "$Reg \,"
                        + str(value[0].title())
                        + r"\,\lambda=\,"
                        + str(value[1])
                        + "$, "
                    )
            case C.L_DIMBATCH:
                caption += r"$Batch size:\," + str(value) + "$ "
            case C.L_MOMENTUM:
                if value[0] != False:
                    caption += (
                        "$Momentum:\, "
                        + value[0].title()
                        + r"\,\alpha: "
                        + str(value[1])
                        + "$, "
                    )
            case C.L_EPOCHS:
                caption += r"$MaxEpochs:\," + str(value) + "$, "
            case C.L_EPS:
                caption += r"$\epsilon:\," + str(value) + "$, "
            case C.L_DISTRIBUTION:
                caption += f"${str(value)}$, "
            case C.L_BIAS:
                caption += r"$Bias:\," + str(value) + "$, "
            case C.L_EARLYSTOP:
                if value:
                    caption += (
                        r"$EarlyStop:\,"
                        + str(theta["treshold_variance"])
                        + "$, "
                    )
            case C.L_PATIENCE:
                caption += r"$Patience:\," + str(value) + "$ "

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
def plot_curves(loss_tr, loss_vs, metr_tr, val_metr, error_tr = False, path = None, ylim = (0., 10.), yMSElim = (0, 0), lbl_tr:str = C.LABEL_PLOT_TRAINING, lbl_vs:str = C.LABEL_PLOT_VALIDATION, titlePlot:str = '', theta: dict = None, labelsX: list[str] = None, labelsY: list[str] = None):
    if theta is None:
        theta = {}
    if labelsX is None:
        labelsX = ['Epochs','Epochs']
    if labelsY is None:
        labelsY = ['Loss','Metric']
    proc=mp.Process(target=draw, args=(loss_tr, loss_vs, metr_tr, val_metr,error_tr, path, ylim ,yMSElim, lbl_tr,lbl_vs, titlePlot, theta, labelsX, labelsY))
    proc.daemon=False
    proc.start()
    proc.join()
    

def draw_async(*args, **kwargs):
    # Creare un processo e passare la funzione draw come target
    draw_process = mp.Process(target=draw, args=args, kwargs=kwargs)

    # Avviare il processo
    draw_process.start()

    # Attendere la fine del processo (opzionale)
    # draw_process.join()



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
def draw(loss_tr, loss_vs, metr_tr, val_metr, error_tr = False, path = None, ylim = (0., 10.), yMSElim = (0, 0), lbl_tr:str = C.LABEL_PLOT_TRAINING, lbl_vs:str = C.LABEL_PLOT_VALIDATION, titlePlot:str = '', theta: dict = None, labelsX: list[str] = None, labelsY: list[str] = None):
    
    if theta is None:
        theta = {}
    if labelsX is None:
        labelsX = ['Epochs','Epochs']
    if labelsY is None:
        labelsY = ['Loss','Metric']
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
    if error_tr and C.PLOT_LOSS:
        ax[0].plot(range(len(error_tr)), error_tr, color='y', label="Loss TR")

    set_legend(ax, 0, labelsX, labelsY)
    if yMSElim!= (0,0) and (not np.isnan(yMSElim[1])):
        ax[0].set_ylim(yMSElim)

    ax[1].plot(range(len(metr_tr)), metr_tr, color='b', linestyle='dashed', label=lbl_tr)
    ax[1].plot(range(len(val_metr)), val_metr, color='r', label=lbl_vs)
    set_legend(ax, 1, labelsX, labelsY)
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


def set_legend(ax, arg1, labelsX, labelsY):
    ax[arg1].legend(loc='best', prop={'size': 9})
    ax[arg1].set_xlabel(labelsX[arg1], fontweight='bold')
    #ax[0].set_xlabel(labelsX[0] + "\\*" + caption ,fontweight='bold')
    ax[arg1].set_ylabel(labelsY[arg1], fontweight='bold')