
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
  --source-image-family=ubuntu-2004-lts-a100 \
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
  --network-interface subnet-name=default,nat-ip-version=ipv4 \
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
  Hostname 62.84.118.238
  User yc-user
```

Install CUDA runtime. Required by bitsandbytes

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-7
```

Fix A100 Xid 119 https://github.com/NVIDIA/open-gpu-kernel-modules/issues/446#issuecomment-1476504064

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

Specific Torch version

```
pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
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

Sync code and data

```
scp tasks/* rulm-sbs:~/tasks
scp evals/* rulm-sbs:~/evals
scp main.* requirements.txt rulm-sbs:~

scp 'rulm-sbs:~/evals/*' evals
scp 'rulm-sbs:~/main.*' .
```
