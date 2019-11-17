# PROGRAMMER: Minglun Zhu
# DATE CREATED: 2019-11-04

#    Basic usage: python train.py data_directory
#       the data_directory must have the correct hierarchy and include training, validation and testing
#       named like this: train, valid, test
#    Prints out training loss, validation loss, and validation accuracy as the network trains
#    Options:
#        Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#        Choose architecture: python train.py data_dir --arch "vgg13"
#        Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#        Use GPU for training: python train.py data_dir --gpu

#import packages
import helper as hpr
from args import Args_Train as AT
import torch
from torch import optim
from torchvision import datasets, transforms
from workspace_utils import keep_awake
from numpy import array as npArr
import pandas as pd
import matplotlib.pyplot as plt

def main():
    #get input args
    iptArgs = AT().parseArgs() 
    
    #train mdl according
    #on data directory
    #the specified nn architecture
    #learning rate, hidden units, epochs
    #if gpu
    #and where to save the checkpoint
    
    #set up data dir and pre process the data
    data_dir = iptArgs.dataDir
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),
        'vldTest': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
        'vld': datasets.ImageFolder(valid_dir, transform = data_transforms['vldTest']),
        'test': datasets.ImageFolder(test_dir, transform = data_transforms['vldTest'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 128, shuffle = True),
        'vld': torch.utils.data.DataLoader(image_datasets['vld'], batch_size = 128),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 128)
    }
    
    #build nn with the specified nn architecture
    #hidden units
    arch = iptArgs.arch
    
    mdl = hpr.buildMdl(
        arch,
        iptArgs.hidden_units
    )
    
    #set learning rate
    # Only train the classifier parameters, feature parameters are frozen
    optmzr = optim.Adam(hpr.MDLS[arch]['getParams'](mdl), lr = iptArgs.learning_rate)
    
    #set device
    dvc = hpr.getDevice(iptArgs.gpu)
    
    #train for epochs
    mdl.to(dvc)

    lossHist_train, lossHist_vld = [], []

    #keep the state_dict for the best model
    #if after new training the vld loss never beat the saved model
    #then the original saved model would just be saved again
    try:
        lowestVldLoss = mdl.lowestVldLoss
    except AttributeError:
        lowestVldLoss = 99.9

    bestSd = mdl.state_dict()

    for e in keep_awake(range(iptArgs.epochs)):
        #reset loss for each epoch
        losses_train, losses_vld = [], []

        mdl.train()

        for imgs, lbls in dataloaders['train']:
            # Move input and label tensors to the GPU
            imgs, lbls = imgs.to(dvc), lbls.to(dvc)

            optmzr.zero_grad()

            logits = mdl.forward(imgs)
            loss = hpr.ce(logits, lbls)
            loss.backward()
            optmzr.step()

            #add up all the train loss for the epoch
            losses_train.append(loss.item())

        #vld pass
        losses_vld, probs, crcts = hpr.vldTest(mdl, dataloaders['vld'], dvc)

        #avg loss per batch for the epoch
        loss_train = npArr(losses_train).mean()
        loss_vld = npArr(losses_vld).mean()
        accuracy_vld = npArr(crcts).mean()

        #add loss to history
        lossHist_train.append({
            'epoch': e,
            'loss': loss_train
        })

        lossHist_vld.append({
            'epoch': e,
            'loss': loss_vld
        })

        #keep track of the lowest vld loss
        if loss_vld < lowestVldLoss:
            mdl.lowestVldLoss = lowestVldLoss = loss_vld

            #keep state_dict of the lowest vld loss model
            bestSd = mdl.state_dict()

        #print out each epoch as to get a sense of progress
        print('epoch: {} / {}, train loss: {}, vld loss: {}, vld accuracy: {}'.format(e + 1, iptArgs.epochs, loss_train, loss_vld, accuracy_vld))

    #save mdl
    torch.save(
        {
            'arch': arch,
            'hnCnt': iptArgs.hidden_units,
            'sd': bestSd,
            'lvl': lowestVldLoss
        },
        iptArgs.save_dir+'/'+hpr.CP_NAME
    )

    #plot loss history
    #can't display charts
    #plt.errorbar(data = pd.DataFrame(lossHist_train), x = 'epoch', y = 'loss')
    #plt.errorbar(data = pd.DataFrame(lossHist_vld), x = 'epoch', y = 'loss')

# Call to main function to run the program
if __name__ == "__main__":
    main()