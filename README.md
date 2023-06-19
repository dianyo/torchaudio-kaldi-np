# torchaudio-kaldi-np

This repository reimplements the [kaldi's algorithm in torchaudio module](https://pytorch.org/audio/stable/compliance.kaldi.html) using numpy. The torch and torchaudio library is designed for research purpose ,and it is too large for the production, especially small disk embedding system, to install the whole dependencies. Although there are alternative solutions to solve this problem, such as convert the whole model including the torchaudio part into ONNX, using numpy might be the first way you'd like to try, as you don't need to modify anything in the model structure itself.

## Test Environment

- Mac 12.5.1 (Apple M1) and Ubuntu 20.04
- python 3.8
- numpy 1.22.2
- torch 2.0.0
- torchaudio 2.0.1

You can run the test script to see if your own version works or not.

```shell
python test.py
```

## Usage

Just copy the script and replace the imported torchaudio module in your code.

```python
# replace this
import torch.compliance.kaldi as kaldi
# with this
import kaldi_np as kaldi
```
