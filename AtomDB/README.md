# AtomDB

Test and collect operator info, build database.

## Setup

### install requirements

```shell
# clone onnx2keras first
git clone https://github.com/gmalivenko/onnx2keras.git submodule/onnx2keras
# install specific onnx2keras and python requirements
bash setup.sh
```

### setup workspaces and stm32 projects

1. install STM32CUBEMX and install F4/F7/H7 packages and X-CUBE-AI 7.2.0 extension.
2. install STM32CUBEIDE and STM32Programmer for open projects and get the board serial number.
3. check stm32 projects in submodule/mcurazer/stm32projs, you will find some preset projects, use STM32CUBEIDE to open these projects all with submodule/mcurazer/workspaces/h743zi as the default workspace, then run setup_workspace.sh to clone all these spaces cause there is a lock when compile projects.

### setup mcudb scripts

1. get the board serial number using STM32Programmer
2. if you use windows + wsl + powershell, you should install python env in wsl first
   1. update the hardcode part in mcudb.ps1 for the correct serial number and path
3. if you use linux
   1. update the udev setting for board tty alias

 ```shell
 sudo vim /etc/udev/rules.d/98-stmcom.rules
 # add com_udev.rules
 sudo service udev reload && sudo service udev restart
 # reboot
 ```

   1. update the hardcode part in mcudb.sh for the correct serial number and path

## Usage

### test model

```shell
# windows
./mcudb.ps1 --board h743zi --cmd validate --model <model_path>
# linux
./mcudb.sh -b h743zi -n 0 -m <model_path>
# linux test many models continuously
ls | xargs -n1 -i echo './mcudb.sh -b h743zi -n 0 -m {}' | bash
```

### parse stm32ai log

```shell
python utils/mcuprofiler.py -d <log_root_dir>
```

### generate AtomDB CSV format

```shell
python utils/generate_atomdb.py -d <log_root_dir> -n <output_filename>
```
