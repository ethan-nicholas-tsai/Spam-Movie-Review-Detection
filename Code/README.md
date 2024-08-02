# Get Started
## Installation

```shell
conda create -n spam python=3.8

conda activate spam /  source activate spam

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch && pip install pyg-lib torch_sparse -f https://data.pyg.org/whl/torch-1.12.1+cu102.html && pip install torch-geometric==2.3.1

git clone https://github.com/ethan-nicholas-tsai/Spam-Movie-Review-Detection.git

<!-- git clone https://gitee.com/cendeavor/spam-movie-review-detection.git -->

cd spam-movie-review-detection

git branch -a  # Look-up all branches

git checkout -b newest origin/newest  # Locally create a new branch named 'newest' and set it up to track the remote 'origin/newest' branch.

git branch # Look-up the current branch

pip install -r requirements.txt

<!-- git clone https://gitee.com/cendeavor/deep-learning-experiment-platform.git experiments -->

<!-- cd experiments && pip install -r requirements.txt -->

cd .. && cd models && cd pretrained && mkdir BERT && cd BERT && Upload the fine-tuned pre-trained model (TODO: Upload to a repository and pull it down)

cd ../../../exps/newest && mkdir cfgs && cd cfgs && Upload the configuration file (alternatively, you can directly use a config file to generate a template and fill it in, TODO)

cd ../../../ && touch main_exp.py main_test.py main_play.py
```

### Future development
Git branch switching：https://www.cnblogs.com/smiler/p/6924583.html
cd .git
git config user.name "xxx"
git config user.email "xxx@xxx"
git branch --set-upstream-to=origin/newest newest
cd ..

### Other
Compiling torch-sparse can be quite slow, so it is recommended to choose a high-performance CPU server with a large number of cores. Additionally, when compiling, it is crucial to ensure that the local CUDA and GCC versions are accurate, as incorrect versions may prevent successful compilation.

## Recommended Configuration
Python 3.8
Pytorch 12.1, Cuda 10.2/11.3, GCC 7+
PyG 2.3.1+ (2.4.0+，HeterModel runs faster)
torch_sparse 0.6.13+

Theoretically, this code can run on Python 3.8+, PyTorch 1.8+, and PyTorch Geometric (PyG) 2.3.1+ (tested). We have tested it on NVIDIA 3090 GPUs and Tesla V100 GPUs

**Note：**
1. "When loading a model for prediction, the library versions used for training that model must be exactly the same!"
2. "The NVIDIA GeForce RTX 3090 GPU only supports CUDA 11 and above versions, while the GeForce RTX 3070 can still support CUDA 10.2."

## Instructions for Installing Multiple CUDA Versions for Multi-User Devices
Refer to https://www.autodl.com/docs/cuda/

CUDA Priority for Usage:
1. Native CUDA and cuDNN installation
2. CUDA toolkits installed via PyTorch
3. Manually installed CUDA toolkits

If Python dependency libraries require CUDA compilation, such as during installation via wheel (whl) files, it is necessary to configure the corresponding versions of CUDA and cuDNN.

First, `vim .cudaxxx`
`export PATH=/root/cuda-10.2/bin:$PATH`
`export LD_LIBRARY_PATH=/root/cuda-10.2/lib64:$LD_LIBRARY_PATH`

Then, `source .cudaxxx`，After installing the corresponding version of the dependent libraries, you should be able to successfully compile. 

Note: The modified $PATH entries should always be appended to the end, not prepended, to avoid accidentally reading the native CUDA installation by default. You can use nvcc -V to verify the CUDA version being used.

## Installation Instructions for GCC for Regular Users
https://zhuanlan.zhihu.com/p/583630190