# AudioSet VGGish in PyTorch


## Introduction
This repository includes:
- A script which converts the pretrained VGGish model provided in the AudioSet repository from TensorFlow to PyTorch
(along with a basic smoke test).  
**Sourced from:** https://github.com/tensorflow/models/tree/master/research/audioset
- The VGGish architecture defined in PyTorch.  
**Adapted from:** https://github.com/harritaylor/torchvggish
- The converted weights found in the [Releases](https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch/releases) section.

Please note that converted model does not produce exactly the same results as the original model, but should be 
close in most cases.

## Usage
1. Download the pretrained weights and PCA parameters from the [AudioSet](https://github.com/tensorflow/models/tree/master/research/audioset) repository and place them in the working directory. 
2. Install any dependencies required by [AudioSet](https://github.com/tensorflow/models/tree/master/research/audioset) (e.g., resampy, numpy, TensorFlow, etc.).
3. Run **"convert_to_pytorch.py"** to generate the PyTorch formatted weights for the VGGish model or download
the weights from the [Releases](https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch/releases) section.

## Example Usage
Please refer to the **"example_usage.py"** script. The output of the script should be as follows.

```
Input Shape: (3, 1, 96, 64)
Output Shape: (3, 128)
Computed Embedding Mean and Standard Deviation: 0.13079901 0.23851949
Expected Embedding Mean and Standard Deviation: 0.131 0.238
Computed Post-processed Embedding Mean and Standard Deviation: 123.01041666666667 75.51479501722199
Expected Post-processed Embedding Mean and Standard Deviation: 123.0 75.0
```