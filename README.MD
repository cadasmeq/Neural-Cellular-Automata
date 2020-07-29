# # Growning Neural Cellular Automata (Python)
Based en in the [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) of [distill.pub](https://distill.pub/). I started from scratch a version of __Neural Cellular Automata__ [__NCA__] using ```Pytorch``` as Framework for Machine Learning.

This project started as a way to practice my  python  skills after to watched a Spanish Youtuber named [DotTCSV](https://www.youtube.com/dotcsv) who coded the same project from scratch. 
He stopped in the *Experiment 1: Learning to Grow*, but I would like to finish it and keep testing this kind of model in future.

# # Starting 🚀 
### What contains this Repo?
This repository have a model that allows you train a Neural Cellular Automata such is explained in [original post](https://distill.pub/2020/growing-ca/), __BUT__ until the "Experiment 1".

### Observations! 👀
At the current version, this repo have a lot of dirty code that eventually will be cleaned. Also,  Model have a few functional issues during the training process that I would like to investigate indeep if those correspond to the main architecture or just simply it's me failing in fine tunning the model's details. I'll add these observations as a topic in TO:DO if you want to have any knowledge of them. Once fixed it, I'll continue with rest of Experiments

# # Pre-requisites 📋
I used  ```Python 3.7 ``` and the next list of packages:
```
numpy==1.18.1
matplotlib==3.1.3
torch==1.5.1
torchvision==0.6.1
opencv-python==4.3.0.36
pillow==7.2.0
```
Also ```/input/image.png```  of size ```32x32``` with without solid color in background (transparent).

# # Train NCA 🔧

To start training:
```
$ python main.py
```
and after a specified number of ```Epochs```, will plot the train result.


# # Train NCA (Notebook) 🔧 
This repo also contains a notebook that contains whole code to quick run in a single cell named ```NCA-Notebook.ipynb``` only is needed upload the target picture and set its path (same as before).

# # TODO (Short-Term)
 - ~~Enabling Git.~~
 - Export and save plots of the training step.
 - Save and load model's weights after train.
 - Add an evaluation function.
 - Test ```perception``` function and check differences with output of ```perception_v2```.
 - Clean commenteds notes and redundant code.
 - Add a dynamical way to select cuda/cpu.
 - Test another hyperparameters and check the variance of results.
 - Test way to droput model when board is empty at start.

# # TODO (Long-Term):
 - Code Experiment 2 and Experiment 3 of the original post.