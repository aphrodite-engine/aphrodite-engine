---
title: Windows Installation
---

Aphrodite Engine supports native Windows builds, alongside Windows Subsystem for Linux (WSL) builds. This guide will cover both methods of installation.


Generally, it's recommended to use WSL, as it seems to be slightly faster than native Windows when batched - and it's easier to set up if you're building from source, or wish to use Triton functionalities, such as LoRA and guided decoding.
***
## Native Windows

We recommend CUDA 12.4.1 for Windows. You can download it from [here](https://developer.nvidia.com/cuda-12-4-1-download-archive).

### Installing from PyPI

First, make sure you have Python installed. You can confirm this by searching for `cmd` in the Start menu, opening it, and running `python --version`. If it returns 3.8 or newer, you're good. If not, you will need to install Python. See [here](https://phoenixnap.com/kb/how-to-install-python-3-windows) for a detailed guide. Next, you will need to create a virtual environment. Search for "PowerShell" in the Start menu, open it, and run the following commands:

```console
python -m venv venv --prompt aphrodite

# Activate the virtual environment
venv\Scripts\activate
```

Please run the last command every time you open a new PowerShell window!

You will need to separately download PyTorch, as the default Windows build is made for CPU only.

```console
pip install 'torch==2.4.1' --index-url https://download.pytorch.org/whl/cu124
```

Then, install Xformers:

```console
pip install https://downloads.pygmalion.chat/whl/windows/xformers/xformers-0.0.28-cp312-cp312-win_amd64.whl
```

Please replace the `cp312-cp312` with your Python version, which you can find by running `python --version`. For example, if you have Python 3.10, you should replace it with `cp310-cp310`.

Finally, install Aphrodite Engine:

```console
pip install aphrodite-engine
```

And you're done! You can now use Aphrodite Engine by running `aphrodite run <your model path or HF ID>` in the PowerShell window. Run `aphrodite run --help` for more information. Make sure you activate the virtual environment every time you open a new PowerShell window!

### Building from Source

This is a lot more complicated than installing from PyPI, and generally not recommended unless you know what you're doing. Below is a very compressed guide on how to build Aphrodite Engine from source on Windows.

#### Prerequisites

##### Visual Studio 2022
Download and install Visual Studio 2022 from [here](https://visualstudio.microsoft.com/downloads/). Download the Community edition, as it's free.

Once the Visual Studio installer is launched, make sure to select the "Desktop development with C++" workload. Next, navigate to the "Individual components" tab and select the following components (use the search bar to find them):

- C++ CMake tools for Windows
- Git for Windows
- C++ Clang Compiler for Windows
- MS-Build support for LLVM-Toolset (clang)

Once you've selected these components, click "Install" and wait for the installation to complete. You may need to restart your computer after the installation is complete (to be confirmed).

##### CUDA 12.4.1
Download and install CUDA 12.4.1 from [here](https://developer.nvidia.com/cuda-12-4-1-download-archive). Make sure to select the correct installer for your system.

#### Building

Simply, clone the repository and run the following commands in `Developer PowerShell for VS 2022` (search for it in the Start menu):

```console
git clone https://github.com/PygmalionAI/aphrodite-engine.git
cd aphrodite-engine

# Create a virtual environment
python -m venv venv --prompt aphrodite
# Activate the virtual environment
venv\Scripts\activate

# Install the required dependencies
pip install https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp312-cp312-win_amd64.whl
pip install https://downloads.pygmalion.chat/whl/windows/xformers/xformers-0.0.28-cp312-cp312-win_amd64.whl

# Build Aphrodite Engine
pip install -e . --no-build-isolation
```

Please replace the `cp312-cp312` with your Python version. For example, if you have Python 3.10, you should replace it with `cp310-cp310`.

Many things can go wrong during the build process. Below is a list of common issues and their solutions:

##### Common Build Issues

- `No CUDA toolset found.`

This happens if you don't have the Visual Studio IDE installed and you're only using the build tools. You can either install the IDE (useless) or perform the fix manually:

1. Copy the 8 files from the following directories:

```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations
```

2. Paste them into the following directory:

```
C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

Now try building again.

- `Failed to find nvToolsExt`

The story behind this error is fascinating. The CUDA installer stopped shipping compiled binaries for nvToolsExt, which is a dependency for PyTorch, since CUDA 12.0. PyTorch never updated their code to use the recommended nvToolsExt headers, so they still rely on the old binaries. This makes building code that uses PyTorch on CUDA 12.0+ a nightmare. The solution seems to be this:

1. Download the NvToolsExt.7z file from the official pytorch builder: https://ossci-windows.s3.us-east-1.amazonaws.com/builder/NvToolsExt.7z

(you can also just install CUDA 11.8 but with only the NSight NVTX component selected)

2. Extract the contents of the archive. You can use a 7zip tool for this, like 7zip or WinRAR.

3. Copy the `NvToolsExt` directory to the following location:

```
C:\ProgramFiles\NVIDIA Corporation
```

You may need to replace the existing `NvToolsExt` directory. Now try building again.


## Windows Subsystem for Linux (WSL)

The installation process for WSL is essentially the same as Linux. You can refer the [NVIDIA GPU Guide](/installation/installation) or the [AMD GPU Guide](/installation/installation-rocm) for more information. Just make sure you've installed CUDA on your Windows machine, and you should be good to go. You can't install CUDA on WSL.
