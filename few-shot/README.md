# Quantum Prototypical Networks for Few-shot Learning

This is a modification of original Prototypical Networks for Few-shot Learning.

The following modifications have been added on top of the original model:

+ We applied qtensor into the model and replace the ....

+ We added miniImagenet data for benchmarking

+ We conducted an end-to-end training and serving pipeline for quantum few-shot learning

+ We accelerated the training and serving process for quantum few-shot learning in light of CUDA graph

+ We imported medical dataset to the pipeline as an application showcase

## Training a quantum prototypical network

### Install dependencies

* This code has been tested on Ubuntu 20.04 with Python 3.8 and PyTorch 1.10.1
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install [torchnet](https://github.com/pytorch/tnt) by `pip install torchnet` or running `pip install git+https://github.com/pytorch/tnt.git@master`.
* Install the quantum protonets package by running `python setup.py install`.

### QTensor Installation ###

First, you need to install QTensor from [source code](https://github.com/danlkv/qtensor).

```bash
# --recurse-submodules is important since qtensor has a submodule qtree  
git clone --recurse-submodules https://github.com/DaniloZZZ/QTensor
cd QTensor

# using git branch -v -a to check all branchs
# and switch to dev branch
git switch dev
# checkout the specific commit 
git checkout dc53509

cd qtree
pip install .

cd .. # back to QTensor folder
pip install .
```

You can check if the `qtensor` and `qtree` are installed or not using `pip list`

### QTensorAI Installation ###

```bash
cd QTensorAI
python3 setup.py install
```

You can check if the `qtensor-ai` is installed or not using `pip list`


### Set up the dataset

* Run `sh download_omniglot.sh`.

* Run `sh download_miniImagenet.sh`


### Train the model (including evaluation)

For the Omniglot dataset, run `python scripts/train/few_shot/run_train.py --data.cuda`. This will run training on GPU and place the results into `results`. For all the options, please use `--help`.

For the miniImageNet dataset, run `python miniImageNet_ext/train_eval.py`. This will run training on GPU and place the results into `results`. For all the options, please use `--help`.
  

### Test the mode

For the Omniglot dataset, run evaluation as: `python scripts/predict/few_shot/run_eval.py --model.model_path results/trainval/best_model.pt`. This will test the best model you have trained.

For the miniImageNet dataset, run `python miniImageNet_ext/test.py`. This will test the best model you have trained, which is stored as `miniImageNet_ext/results/best_model.pth`.
