# Requirements 
To run this program, you need install at least
- Pytorch
- Numpy (might be automatically install with Pytorch)
- tqdm (for progress bar)
  
Other things might be require. Install the modules suit you environment.

# Original Paper
[Pixel Objectness](http://vision.cs.utexas.edu/projects/pixelobjectness/)  
  
# Dataset
You need to prepare your dataset.  
Download the dataset from
- [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html)  
  
# Preprocess
First, you need to preprocess the dataset to make binalized mask image.
```
python misc/create_dataset.py --voc_root PATH_TO/VOCdevkit/ --sbd_root PATH_TO/SBDdataset/benchmark_RELEASE
```
This will do the all things and, will make train.pkl and val.pkl which are pickled dataset.

# Train
First, you need to download the ImageNet pretrained parameter from pytorch official.
```
wget https://download.pytorch.org/models/vgg16-397923af.pth
```
  
then  
  
```
python train.py
```
There are some options you can change. See the code or use --help for more detatil.  

# Prediction
```
python prediction.py --image_dir folder_that_contains_images/
```
