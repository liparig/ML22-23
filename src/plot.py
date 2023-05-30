import matplotlib as plt

def plot_curves(loss_tr, loss_vs, metr_tr, val_metr, path=None, ylim=(0., 10.), lbl_tr='development',
                lbl_vs='internal test', *args):
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
    figure, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(len(loss_tr)), loss_tr, color='b', linestyle='dashed', label=lbl_tr)
    ax[0].plot(range(len(loss_vs)), loss_vs, color='r', label=lbl_vs)
    ax[0].legend(loc='best', prop={'size': 9})
    ax[0].set_xlabel('Epochs', fontweight='bold')
    ax[0].set_ylabel('Loss', fontweight='bold')
    ax[0].grid()
    ax[1].plot(range(len(metr_tr)), metr_tr, color='b', linestyle='dashed', label=lbl_tr)
    ax[1].plot(range(len(val_metr)), val_metr, color='r', label=lbl_vs)
    ax[1].legend(loc='best', prop={'size': 9})
    ax[1].set_xlabel('Epochs', fontweight='bold')
    ax[1].set_ylabel('Metric', fontweight='bold')
    ax[1].set_ylim(ylim)
    ax[1].grid()
    
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
