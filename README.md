# Requirements 
To run this program, you need install at least
- Pytorch
- Numpy (might be automatically install with Pytorch)
- tqdm (for progress bar)
  
Other things might be needed, install the modules suit you environment.

# Paper
[site](http://vision.cs.utexas.edu/projects/pixelobjectness/)  
  
# Dataset
You need to prepare your dataset.  
Download the dataset from
- [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html)  
  
The default dataset's directory configuration is
```
dataset/
       |- sbd/
       |     |- img/      #images
       |     |- mask/     #labels
       |     |- train.txt
       |     |- val.txt
       |
       |- voc/
             |- img/      #images
             |- mask/     #labels
	     |- train.txt
	     |- trainval.txt
	     |- val.txt
```
SBD dataset and VOC dataset has same images, so you can marge the directories if you want.  
train.txt and so on things are files that are written which image to use. These are come with the dataset  
  
## Binarize class
Before staring a training, we need to binarize tha class labels to foreground and background.  
I made a script for doing this job. See the misc directory.  
- convert2binary.py is for binalizing class labels which convert 0 to background and 1~ to foreground.
- mat2omg.py is for converting mat files to image files, default is .png format.

# Train
```
python train.py
```
There are some options you can change. See the code or use --help for more detatil.  

# Prediction
```
python prediction.py
```
