import matplotlib.pyplot as plt
import datetime

from costants import FORMATTIMESTAMP
class dnn_plot:
    
    def __init__(self,x:list,y:list,plotnumber:int,row:int,column:int,plots_lbl:list,xlabel:str,ylabel:str,path:str=None, ylim:tuple=(0., 5.),figsize:tuple=(12,4),title:str=''):
        self.x=x
        self.y=y
        self.plotnumber=plotnumber
        self.row=row,
        self.column=column
        self.plots_lbl=plots_lbl
        self.title=title
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.ylim=ylim
        self.path=path
    
    def _setDefaultDNNPlot(self):
        self.xlabel=['Epochs','Epochs']
        self.ylabel=['Loss','Metric']
        self.plotnumber=2
        self.row=1
        self.column=2
        self.plots_lbl=[('Training','Validation'),('Training','Validation')]
        self.title='Learning'
    
    def plot_curves(self):
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
        plt.close()
        figure, ax = plt.subplots(1, 2, figsize = (12, 4))
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
            s = f'{path}{datetime.datetime.now().strftime(FORMATTIMESTAMP)}.jpg'
            plt.savefig(s)