# PROGRAMMER: Minglun Zhu
# DATE CREATED: 2019-11-04

#    Basic usage: python predict.py /path/to/image checkpoint
#    Options:
#        Return top K most likely classes: python predict.py input checkpoint --top_k 3
#        Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#        Use GPU for inference: python predict.py input checkpoint --gpu

#import packages
import helper as hpr
from args import Args_Pred as AP
from PIL import Image
import torch
import torch.nn.functional as F
import json

def main():
    #get input args
    iptArgs = AP().parseArgs() 
    
    #load and process image
    img = Image.open(iptArgs.imgPath)
    
    #apply transforms
    imgData = hpr.procImg(img)
    
    #to tensor
    #needs to be float tensor
    img = torch.from_numpy(imgData).float()
    
    #load mdl
    mdl = hpr.ldMdl(iptArgs.cpPath)
    
    #get probs
    with torch.no_grad():
        mdl.eval()
        
        #img to device
        img.to(hpr.getDevice(iptArgs.gpu))
        
        #since we only have 1 image, the batch dimension is missing
        #which the model requires (why this is not explained)
        #so we need to use unsqueeze to add the batch dimension at position 0
        #I don't understand why this was never explained
        #and I had to debug forever to find out
        logits = mdl(img.unsqueeze_(0))
            
        probs = F.softmax(logits, dim = 1)
    
    #get top k
    ps, cs = probs.topk(iptArgs.top_k, dim = 1)

    #get cls names dictionary
    with open(iptArgs.category_names, 'r') as f:
        clsNames = json.load(f)

    #sort clsNames in correct order
    sortedClsNames = sorted(
        clsNames.items(), 
        key = lambda itm: 
            itm[0]
    )
    
    #get clsNames list and probs list
    top_clss = [sortedClsNames[i][1] for i in cs.tolist()[0]]
    top_probs = ps.tolist()[0]

    print(top_clss)
    print(top_probs)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()