# PROGRAMMER: Minglun Zhu
# DATE CREATED: 2019-11-04

#a few functions to be used repeatedly

#packages
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from numpy import array as npArr

#consts
CP_NAME = 'cp.pth'

def MOD_RES(mdl, fcs):
    mdl.fc = fcs

def MOD_DENSE(mdl, fcs):
    mdl.classifier = fcs

def PARAMS_RES(mdl):
    return mdl.fc.parameters()

def PARAMS_DENSE(mdl):
    return mdl.classifier.parameters()

MDLS = {
    'resnet': {
        'mdl': models.resnet152(pretrained = True),
        'iptUnitCnt': 2048,
        'modFc': MOD_RES,
        'getParams': PARAMS_RES
    },
    'densenet': {
        'mdl': models.densenet161(pretrained = True),
        'iptUnitCnt': 2208,
        'modFc': MOD_DENSE,
        'getParams': PARAMS_DENSE
    }
}
#end consts

ce = nn.CrossEntropyLoss()

def getDevice(gpu):
    d = 'cpu'
    
    if gpu:
        if torch.cuda.is_available():
            d = 'cuda'
        else:
            print('Warning: GPU not available. CPU will be used.')

    return torch.device(d)

def buildMdl(arch, hnCnt):
    mdl = MDLS[arch]['mdl']

    # Freeze parameters so we don't backprop through them
    for param in mdl.parameters():
        param.requires_grad = False

    ic = MDLS[arch]['iptUnitCnt']
    
    fcs = OrderedDict([
        ('fc1', nn.Linear(ic, ic * 2)),
        ('relu1', nn.ReLU()),

        ('fc2', nn.Linear(ic * 2, ic * 4)),
        ('relu2', nn.ReLU()),

        ('fc3', nn.Linear(ic * 4, ic * 2)),
        ('relu3', nn.ReLU()),

        ('fc4', nn.Linear(ic * 2, ic)),
        ('relu4', nn.ReLU()),

        ('fc5', nn.Linear(ic, hnCnt)),
        ('relu5', nn.ReLU()),
        ('do5', nn.Dropout(.2)),

        ('fc6', nn.Linear(hnCnt, hnCnt)),
        ('relu6', nn.ReLU()),
        ('do6', nn.Dropout(.2)),

        ('fc7', nn.Linear(hnCnt, hnCnt)),
        ('relu7', nn.ReLU()),
        ('do7', nn.Dropout(.2)),

        ('fc8', nn.Linear(hnCnt, hnCnt)),
        ('relu8', nn.ReLU()),
        ('do8', nn.Dropout(.2)),

        ('opt', nn.Linear(hnCnt, 102))
    ])
        
    classifier = nn.Sequential(fcs)
    
    MDLS[arch]['modFc'](mdl, classifier)
    
    return mdl

def vldTest(mdl, dataLdr, device):
    '''
        runs a forward pass on the data loader
        
        params:
            mdl(pytorch model): the model will be used to make predictions
            dataLdr(pytorch data loader): the data loader is expected to have images and labels
                so this function should only be used on validation or testing since you must have labels
                on the data you want to prediction
        returns:
            lossPerB(list): containing validation or test loss from each batch
            probsPerB(list): containing softmax probabilities for each image each batch
                each list item is a batch which has rows and columns
                rows corresponds to images, and each column represents the probability for a class
            corrects(list): list of 1s and 0s indicating 1 as correct prediction, and 0 as incorrect
    '''
    lossPerB = []
    probsPerB = []
    corrects = []
    
    # turn off gradients
    with torch.no_grad():
        # set model to evaluation mode so that dropout doesn't get applied
        mdl.eval()
        
        # validation pass here
        for imgs, lbls in dataLdr:
            # Move input and label tensors to the GPU
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            logits = mdl(imgs)
            
            loss = ce(logits, lbls)
            
            probs = F.softmax(logits, dim = 1)
            
            clss = probs.topk(1, dim = 1)[1]
            
            corrects.extend(clss == lbls.view(*clss.shape))
            
            #add up all the vld loss for each epoch
            lossPerB.append(loss.item())
            #add up all the probs
            probsPerB.append(probs)
            
    return lossPerB, probsPerB, corrects

def buildResnet152(fcs):
    mdl = MDLS['resnet']['mdl']

    # Freeze parameters so we don't backprop through them
    for param in mdl.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(fcs)
    
    mdl.fc = classifier
    
    return mdl

def ldMdl(f):
    #extra arguments for loading with cpu
    cp = torch.load(f, map_location = lambda storage, loc:storage)
    
    #mdl from part1 has different settings saved
    if (f == 'part1Cp/cp.pth'):
        mdl = buildResnet152(cp['fcs'])
    else:
        mdl = buildMdl(cp['arch'], cp['hnCnt'])

    mdl.load_state_dict(cp['sd'])
    
    mdl.lowestVldLoss = cp['lvl']
    
    return mdl

def procImg(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #resize
    #shortest side = 256
    w, h = image.width, image.height
    
    ss = min(w, h)
    
    dlta_ratio = 256 / ss
    
    #modified in place
    image.thumbnail((w * dlta_ratio, h * dlta_ratio))
    
    #center crop
    wc, hc = image.width / 2, image.height / 2
    
    top = hc - 112
    lft = wc - 112
    rgt = wc + 112
    btm = hc + 112
    
    img_crop = image.crop((lft, top, rgt, btm))
    
    #normalize
    img_np = npArr(img_crop)
    
    #we need to convert int arr to float arr first
    #because we want the result to be float arr
    img_np = img_np.astype('float64') / 255
    
    img_np = (img_np - npArr([0.485, 0.456, 0.406])) / npArr([0.229, 0.224, 0.225])
    
    img_t = img_np.transpose((2, 0, 1))
    
    return img_t