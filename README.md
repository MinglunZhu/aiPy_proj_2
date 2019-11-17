# aiPy_proj_2

## Description
This is a project I did for Udacity Course AI Programming with Python Project 2 - Create Your Own Image Classifier.

They say create your own, but really you just apply transfer learning to pre-trained ImageNet image classifier. Which granted, is the smarter way because it's a lot faster and probably a lot more accurate too than having to train your own CNN from the ground up.

***Please note: This is not a working demo, it is example code mainly for you to read through.
The code is written to work with Udacity workspaces. Specifically, the training code includes code to keep Udacity workspace awake during training. You'll have to remove this code if you are not using Udacity workspace.***

### Part 1
The project contains two parts. The first part is a jupyter notebook containing code I used to reconfigure and train the pre-trained ResNet152 model. The recongifured model is then trained on the flowers dataset on GPU with 85% accuracy at the end of the first training pass. The model is then saved into a checkpoint, but the checkpoint is 1GB big, so it's not included in this repository.

### Part 2
The second part is a commandline program that allows you to reconfigure a pre-trained model (DenseNet or ResNet) using transfer learning and then train the transfer learning model on specified images directory. Then the program saves the model into a checkpoint.

You can then make prediction by specifying the saved checkpoint you want to use and the image you want to make predictions on.

