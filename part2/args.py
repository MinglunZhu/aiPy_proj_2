# PROGRAMMER: Minglun Zhu
# DATE CREATED: 2019-11-04

#script for get commandline args for the 2 main scripts

#pkgs
from argparse import ArgumentParser as ap
from helper import MDLS

class Args:
    def __init__(self):
        # Create Parse using ArgumentParser
        self.p = ap()
        
        self.p.add_argument(
            '--gpu',
            action = 'store_true',
            help = 'Enable GPU for training.'
        )

    def parseArgs(self):
        return self.p.parse_args()
    
#    Basic usage: python train.py data_directory
#       the data_directory must have the correct hierarchy and include training, validation and testing
#       named like this: train, valid, test
#    Prints out training loss, validation loss, and validation accuracy as the network trains
#    Options:
#        Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#        Choose architecture: python train.py data_dir --arch "vgg13"
#        Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#        Use GPU for training: python train.py data_dir --gpu
class Args_Train(Args):
    def __init__(self):
        Args.__init__(self)
        
        self.p.add_argument(
            'dataDir',
            metavar = 'data_dir',
            type = str,
            help = 'Directory for images folder. Must contain train and valid folders. Make sure each class has its own folder. E.g flowers'
        )

        self.p.add_argument(
            '--save_dir',
            type = str,
            default = 'part2Cp',
            help = 'Directory for saving model checkpoint. E.g. part2Cp'
        )

        self.p.add_argument(
            '--arch',
            type = str,
            default = list(MDLS.keys())[0],
            help = 'Architecture of the image classifier of your choice: {}.'.format(', '.join(MDLS.keys()))
        )

        self.p.add_argument(
            '--learning_rate',
            type = float,
            default = .01,
            help = 'Learning rate for the training. E.g .01'
        )

        self.p.add_argument(
            '--hidden_units',
            type = int,
            default = 512,
            help = 'Base number of hidden units for the classifier layers. E.g 512'
        )

        self.p.add_argument(
            '--epochs',
            type = int,
            default = 20,
            help = 'Number of epochs for training. Note the epoch with the lowest validation loss will be saved, not the final epoch. E.g 20'
        )
    
#    Basic usage: python predict.py /path/to/image checkpoint
#    Options:
#        Return top K most likely classes: python predict.py input checkpoint --top_k 3
#        Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#        Use GPU for inference: python predict.py input checkpoint --gpu
class Args_Pred(Args):
    def __init__(self):
        Args.__init__(self)
        
        self.p.add_argument(
            'imgPath',
            metavar = 'input',
            type = str,
            help = 'Path to the image you want to use as input for prediction. E.g flowers/test/10/image_07090.jpg'
        )
        
        self.p.add_argument(
            'cpPath',
            metavar = 'checkpoint',
            type = str,
            help = 'Path to the checkpoint you want to use to load model for prediction. E.g par1Cp/cp.pth'
        )

        self.p.add_argument(
            '--top_k',
            type = int,
            default = 3,
            help = 'Number of top classes predicted. E.g. 3'
        )

        self.p.add_argument(
            '--category_names',
            type = str,
            default = 'cat_to_name.json',
            help = 'Path to the json file you want to use to translate category index to category name. E.g cat_to_name.json'
        )