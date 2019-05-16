# VGGish in PyTorch


## Introduction
This repository includes:
- A script which converts the pretrained VGGish model provided in the AudioSet repository from TensorFlow to PyTorch
along with a basic smoke test.  
**Sourced from:** https://github.com/tensorflow/models/tree/master/research/audioset
- The VGGish architecture defined in PyTorch.  
**Adapted from:** https://github.com/harritaylor/torchvggish

## Usage
1. Download the pretrained weights from the AudioSet repository, and place in the working directory. 
2. Install any dependencies required by AudioSet (e.g., resampy, numpy, TensorFlow, etc.)
3. Run "convert_to_pytorch.py" to generate the PyTorch formatted weights for the VGGish model.