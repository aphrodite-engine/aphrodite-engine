---
title: Installation with Neuron
---

Aphrodite supports inference with AWS Trainium/Inferentia chips. At the moment Paged Attention is not supported in Neuron SDK, but naive continuous batching is supported in transformers-neuronx. Data types currently supported in Neuron SDK are FP16 and BF16.

## Requirements
- Linux
- Python 3.8 - 3.11
- Accelerator: NeuronCore_v2 (in trn1/inf2 instances)
- PyTorch 2.0.1/2.1.1
- AWS Neuron SDK 2.16/2.17

## Building from Source
The following instructions are for Neuron SDK 2.16 and above.

### Launch Trn1/Inf2 instances
Here are the steps to launch trn1/inf2 instances, in order to install [PyTorch Neuron (“torch-neuronx”) Setup on Ubuntu 22.04 LTS](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html).

- Follow the instructions at [launch an Amazon EC2 Instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance).
- Refer to these pages for more info about instance sizes and pricing: [Trainium1](https://aws.amazon.com/ec2/instance-types/trn1/), [Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/).
- Select Ubuntu Server 22.02 TLS AMI.
- When launching, adjust your primary EBS volume size to a minimum of 512GB.
- After launching, follow the instructions in [Connect to your instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html).

### Install drivers and tools
If [Deep Learning AMI Neuron](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html) is installed, this step is unnecessary. Otherwise, follow this:

```sh
# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages
sudo apt-get update -y

# Install OS headers
sudo apt-get install linux-headers-$(uname -r) -y

# Install git
sudo apt-get install git -y

# install Neuron Driver
sudo apt-get install aws-neuronx-dkms=2.* -y

# Install Neuron Runtime
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Install Neuron Tools
sudo apt-get install aws-neuronx-tools=2.* -y

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH
```

### Install transformers-neuronx
The backend we use for inference is [transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx). Follow the instructions below to install it:

```sh
# Install Python venv
sudo apt-get install -y python3.10-venv g++

# Create Python venv
python3.10 -m venv aws_neuron_venv_pytorch

# Activate Python venv
source aws_neuron_venv_pytorch/bin/activate

# Install Jupyter notebook kernel
pip install ipykernel
python3.10 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli
python -m pip install wget
python -m pip install awscli

# Update Neuron Compiler and Framework
python -m pip install --upgrade neuronx-cc==2.* --pre torch-neuronx==2.1.* torchvision transformers-neuronx
```

### Install Aphrodite from Source
```sh
git clone https://github.com/PygmalionAI/aphrodite-engine.git
cd aphrodite-engine
pip install -U -r requirements-neuron.txt
APHRODITE_TARGET_DEVICE="neuron" pip install -e .
```