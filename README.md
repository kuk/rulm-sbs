
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
yc vpc address create \
  --name default \
  --external-ipv4 zone=ru-central1-a \
  --folder-name rulm-sbs

address=$(yc vpc address get \
  --name default \
  --folder-name rulm-sbs \
  --format json \
  | jq -r .external_ipv4_address.address)

yc compute image list --folder-id standard-images --format json | grep a100 | sort
# ubuntu-2204-lts
# ubuntu-2004-lts-gpu
# ubuntu-2004-lts-a100

yc compute disk create \
  --name default \
  --source-image-name=ubuntu-20-04-lts-gpu-a100-v20230410 \
  --source-image-folder-id=standard-images \
  --type=network-ssd-nonreplicated \
  --size=186 \
  --zone ru-central1-a \
  --folder-name rulm-sbs

# https://cloud.yandex.ru/docs/compute/concepts/vm-platforms
# standard-v3-t4 
# gpu-standard-v2 NVIDIA® Tesla® V100 Intel Xeon Gold 6230
# gpu-standard-v3 NVIDIA® Ampere® A100

# V100
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

# LLM.int8() requires Turing or Ampere GPUs.
# A100
yc compute instance create \
  --name default \
  --public-address $address \
  --use-boot-disk disk-name=default \
  --platform gpu-standard-v3  \
  --gpus=1 \
  --cores=28 \
  --memory=119 \
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
  Hostname $address
  User yc-user
```

Fix A100 Xid 119 https://github.com/NVIDIA/open-gpu-kernel-modules/issues/446#issuecomment-1476504064? Seams to work in ubuntu-20-04-lts-gpu-a100-v20230410

```
sudo nano /etc/modprobe.d/nvidia-gsp.conf
options nvidia NVreg_EnableGpuFirmware=0

sudo update-initramfs -u
sudo reboot

cat /proc/driver/nvidia/params | grep EnableGpuFirmware
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

CUDA + bitsandbytes

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers-515 cuda-toolkit-11-7

# https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md
git clone https://github.com/TimDettmers/bitsandbytes.git
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda-11.7" >> ~/.bashrc
source ~/.bashrc

make cuda11x CUDA_VERSION=117
CUDA_VERSION=117 python setup.py install
```

Specific Torch version

```
pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Other pip reqs

```
pip install \
  matplotlib \
  tqdm \
  pandas \
  openpyxl \
  openai \
  sentencepiece \
  accelerate \
  git+https://github.com/huggingface/transformers.git \
  git+https://github.com/huggingface/peft.git
```

Setup Jupyter

```
pip install jupyter

screen -S jupyter
screen -r jupyter

# ok to have installation open to the world
jupyter notebook \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888 \
  --NotebookApp.token='' \
  --NotebookApp.password=''
```

Llama.cpp

```
git clone https://github.com/ggerganov/llama.cpp.git
make

sudo apt-get install git-lfs
git clone https://huggingface.co/IlyaGusev/llama_7b_ru_turbo_alpaca_lora_llamacpp
```

Sync code and data

```
scp -r tasks evals rulm-sbs:~
scp main.* rulm-sbs:~

scp -r rulm-sbs:~/evals .
scp 'rulm-sbs:~/main.*' .
```
