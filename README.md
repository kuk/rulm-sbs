
Dev env

```bash
python -m venv ~/.venvs/rulm-sbs
source ~/.venvs/rulm-sbs/bin/activate

pip install -r requirements.txt

pip install ipykernel
python -m ipykernel install --user --name rulm-sbs
```

YC instance

```bash
yc resource-manager folder create --name rulm-sbs

yc vpc network create --name default --folder-name rulm-sbs
yc vpc subnet create \
  --name default \
  --network-name default \
  --range 192.168.0.0/24 \
  --zone ru-central1-a \
  --folder-name rulm-sbs

yc compute image list --folder-id standard-images --format json
# ubuntu-2204-lts
# ubuntu-2004-lts-gpu
# ubuntu-2004-lts-a100

yc compute disk create \
  --name default \
  --source-image-family=ubuntu-2004-lts \
  --source-image-folder-id=standard-images \
  --type=network-ssd-nonreplicated \
  --size=186 \
  --zone ru-central1-a \
  --folder-name rulm-sbs

# https://cloud.yandex.ru/docs/compute/concepts/vm-platforms
# standard-v3-t4 
# gpu-standard-v2 NVIDIA® Tesla® V100 Intel Xeon Gold 6230
# gpu-standard-v3 NVIDIA® Ampere® A100

yc compute instance create \
  --name default \
  --network-interface subnet-name=default,nat-ip-version=ipv4 \
  --use-boot-disk disk-name=default \
  --platform gpu-standard-v2  \
  --gpus=1 \
  --cores=8 \
  --memory=48 \
  --preemptible \
  --ssh-key ~/.ssh/id_rsa.pub \
  --zone ru-central1-a \
  --folder-name rulm-sbs

yc compute instance stop --name default --folder-name rulm-sbs
yc compute instance start --name default --folder-name rulm-sbs
yc compute instance delete --name default --folder-name rulm-sbs

yc compute disk delete --name default --folder-name rulm-sbs

yc vpc subnet delete --name default --folder-name rulm-sbs
yc vpc network delete --name default --folder-name rulm-sbs

yc resource-manager folder delete --name rulm-sbs
```

Up `~/.ssh/config`

```
Host rulm-sbs
  Hostname 62.84.118.238
  User yc-user
```

Setup Python env

```
sudo apt update
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y \
  python3.10 \
  python3.10-venv

python3.10 -m venv ~/.env
source ~/.env/bin/activate
```

Setup Jupyter

```
pip install jupyter
sudo apt install -y screen

# ok to have installation open to the world
screen
jupyter notebook \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888 \
  --NotebookApp.token='' \
  --NotebookApp.password=''
```

Setup Nvidia

```
sudo apt-get install nvidia-driver-515-server

# cuda 11.7
# restart

pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Sync code

```
scp -r tasks evals rulm-sbs:~
scp main.* requirements.txt rulm-sbs:~
```


Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching /usr/local/cuda/lib64...
ERROR: /home/yc-user/.env/bin/python3.10: undefined symbol: cudaRuntimeGetVersion
CUDA SETUP: libcudart.so path is None
CUDA SETUP: Is seems that your cuda installation is not in your path. See https://github.com/TimDettmers/bitsandbytes/issues/85 for more information.
CUDA SETUP: CUDA version lower than 11 are currently not supported for LLM.int8(). You will be only to use 8-bit optimizers and quantization routines!!
CUDA SETUP: Highest compute capability among GPUs detected: 7.0
CUDA SETUP: Detected CUDA version 00
CUDA SETUP: Loading binary /home/yc-user/.env/lib/python3.10/site-packages/


libcuda1-340 - NVIDIA CUDA runtime library
boinc-client-nvidia-cuda - metapackage for CUDA-savvy BOINC client and manager
libcuda1-331 - Transitional package for libcuda1-340
libcuda1-331-updates - Transitional package for libcuda1-340
libcuda1-340-updates - Transitional package for libcuda1-340
libcuda1-384 - Transitional package for nvidia-headless-390
libcudart10.1 - NVIDIA CUDA Runtime Library
nvidia-cuda-dev - NVIDIA CUDA development files
nvidia-cuda-doc - NVIDIA CUDA and OpenCL documentation
nvidia-cuda-gdb - NVIDIA CUDA Debugger (GDB)
nvidia-cuda-toolkit - NVIDIA CUDA development toolkit
nvidia-cuda-toolkit-gcc - NVIDIA CUDA development toolkit (GCC compatibility)
python-pycuda-doc - module to access Nvidia‘s CUDA computation API (documentation)
python3-pycuda - Python 3 module to access Nvidia‘s CUDA parallel computation API
python3-pycuda-dbg - Python 3 module to access Nvidia‘s CUDA API (debug extensions)
cuda-drivers-fabricmanager-515 - Meta-package for FM and Driver
cuda-drivers-fabricmanager-450 - Meta-package for FM and Driver
cuda-drivers-fabricmanager-460 - Transitional package for cuda-drivers-fabricmanager-510
cuda-drivers-fabricmanager-470 - Meta-package for FM and Driver
cuda-drivers-fabricmanager-510 - Transitional package for cuda-drivers-fabricmanager-515
cuda-drivers-fabricmanager-525 - Meta-package for FM and Driver

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

sudo apt install cuda-libraries-11-7
