# aiPy_proj_2

## Description
This is a project I did for Udacity Course AI Programming with Python Project 2 - Create Your Own Image Classifier.

They say create your own, but really you just apply transfer learning to pre-trained ImageNet image classifier. Which granted, is the smarter way because it's a lot faster and probably a lot more accurate too than having to train your own CNN from the ground up.

***Please note: This is not a working demo, it is example code mainly for you to read through.
The code is written to work with Udacity workspaces. Specifically, the training code includes code to keep Udacity workspace awake during training. You'll have to remove this code if you are not using Udacity workspace.***

### Part 1
The project contains two parts. The first part is a jupyter notebook containing code I used to reconfigure the pre-trained ResNet152 model. The recongifured model is then trained on the flowers dataset on GPU with 85% accuracy at the end of the first training pass. The model is then saved into a checkpoint, but the checkpoint is 1GB big, so it's not included in this repository.

### Part 2
The second part is a commandline program that allows you to reconfigure a pre-trained model (DenseNet or ResNet) using transfer learning and then train the transfer learning model on specified images directory. Then the program saves the model into a checkpoint.

You can then make prediction by specifying the saved checkpoint you want to use and the image you want to make predictions on.

## How to Use
### Part 1
Part 1 contains a jupyter notebook and an HTML version of the jupyter notebook.
Just download and open the jupyter notebook. Or if you don't have jupyter notebook installed, you can still view it using the HTML version.

### Part 2
Download and run the following in your commandline interface:
```
python train.py --help
```
This will show you the arguments you need to supply for training the neural network.

Once trained, run the following in your commandline interface:
```
python predict.py --help
```
This will tell you what arguments you need to supply to use the prediction script.

Example training directory setup is provided in the `flowers` folder.
You'll notice that the each folder maps to a category index (1, 10, 100, 101, etc.)
You'll also need to supply a json file that maps the category index to category names. An example is provided as `cat_to_name.json`

## Known Issues
- As you'll see in the jupyter notebook, the first training pass works well, but after saving and reloading the model, the second training pass doesn't train the network at all. I think this is because I didn't save and reload the optimizer `state_dict` used for training. It maybe fixable by saving and reloading the optimizer `state_dict`, but I'm no longer working on this project.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
