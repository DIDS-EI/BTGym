# BTGym
A Platform for Behavior Tree Designing and Planning

# 安装


## 安装OmniGibson

```shell
conda create -n omnigibson python=3.10 pytorch=2.5.1 torchvision torchaudio pytorch-cuda=12.4 "numpy<2" -c pytorch -c nvidia
```
注意：
- 安装pytorch-cuda=xxx的版本要和当前机器所用版本一致，如果cuda版本高于12.4，则安装12.4，否则安装兼容的[pytorch](https://pytorch.org/get-started/locally/)版本

在windows下，omnigibson 会自动配置好 isaac-sim 的相关环境变量，所以应该不需要手动配置

## 安装BDDL
BDDL是用来解析Behavior-1K任务的库

直接通过github仓库安装即可
```python
git clone https://github.com/StanfordVL/bddl.git
```


## 安装 cuRobo

```shell
conda activate omnigibson
python -m pip install tomli wheel ninja
git clone https://github.com/NVlabs/curobo.git
cd curobo
python -m pip install -e .[isaacsim] --no-build-isolation
```

测试安装是否成功
```shell
cd BTGym
python examples/curobo_test/fetch_follow_cube.py
```

### windows下curobo安装问题

#### torch 相关报错

需要用到 omni.isaac 下的库，但又需要防止调用 omni.isaac 下的老版本 torch，一个简单办法是删除 omni.isaac 下的 torch 文件夹（或重命名），路径为：
```
~\AppData\Local\ov\pkg\isaac-sim-4.1.0\exts\omni.isaac.ml_archive\pip_prebundle\torch
```

#### gbk 相关报错

控制面板 →  区域 →  管理 →  更改系统区域设置 →  勾选 "Beta版，使用Unicode UTF-8提供全球语言支持"


#### 注意
windows下curobo自带的 examples/isaac_sim/motion_gen_reacher.py 暂未成功运行，但暂时不影响omnigibson中的正常运行


## 安装OMPL
OMPL (the Open Motion Planning Library)，开源运动规划库，由许多基于采样的先进的运动规划算法组成。[官网](https://ompl.kavrakilab.org/download.html)

推荐从源码安装 https://github.com/ompl/ompl.git
按照README安装Boost, CMake, Eigen依赖。

#### 先安装pygccxml和pyplusplus
```
pip install pygccxml pyplusplus
```


### Ubuntu 安装

1. 直接用apt安装依赖
```shell
sudo apt-get install python3-dev libboost-all-dev cmake libeigen3-dev
```


#### 安装OMPL
```
conda activate omnigibson
mkdir -p build/Release
cd build/Release
cmake ../.. -DPYTHON_EXEC=/home/cxl/softwares/anaconda3/envs/omnigibson/bin/python #注意这里一定要把路径改成自己的omnigibson虚拟环境下的python可执行文件路径
make -j 32 update_bindings
make -j 32
```




### Windows 安装 (未成功)

#### Boost

参考[博客](https://blog.csdn.net/qq_38967414/article/details/129347708?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-129347708-blog-141728930.235%5Ev43%5Epc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-129347708-blog-141728930.235%5Ev43%5Epc_blog_bottom_relevance_base3&utm_relevant_index=1)进行安装

1. 在官网下载 [boost_1_82_0-msvc-14.3-64.exe](https://sourceforge.net/projects/boost/files/boost-binaries/1.82.0/boost_1_82_0-msvc-14.3-64.exe/download)
2. 双击安装并记住安装路径

安装 boost.numpy
```
pip install boost.numpy
```

$env:PYTHON_PATH = "C:\Storage\code_external\anaconda3\envs\omnigibson"
$env:BOOST_ROOT = "C:\Storage\code_external\boost_1_82_0"

cmake .. -DPYTHON_INCLUDE_DIRS="$env:PYTHON_PATH\include\python3.10" -DPYTHON_LIBRARIES="$env:PYTHON_PATH\libs\python3.10.lib" -DBOOST_ROOT="$env:BOOST_ROOT" -DBoost_DIR="$env:BOOST_ROOT\lib64-msvc-14.3\cmake\Boost-1.82.0"





#### 安装 CMake
在官网下载最新版本，例如：[cmake-3.30.5-windows-x86_64.msi](https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-windows-x86_64.msi)

验证
```
cmake --version
```


#### 安装 Eigen
1. 在[官网](https://eigen.tuxfamily.org)下载最新版本，例如 [eigen-3.4.0.zip](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip)
2. 解压后将`eigen-3.4.0`目录复制到`C:\Program Files\eigen-3.4.0`
3. 创建build目录并安装（shell需要管理员权限）
```shell
cd eigen-3.4.0
mkdir build && cd build
cmake ..
cmake --install . --prefix "C:/Program Files/Eigen3"
```
3. 添加环境变量:
   - 打开系统环境变量设置
   - 在系统变量中新建 `Eigen3_DIR`，值设为 `C:\Program Files\Eigen3\share\eigen3\cmake`
   - 在系统变量 `CMAKE_PREFIX_PATH` 中添加 `C:\Program Files\Eigen3` (如果没有这个变量就新建)

#### 安装castxml
在 [官网](https://data.kitware.com/#collection/57b5c9e58d777f126827f5a1/folder/57b5de948d777f10f2696370) 下载castxml，例如 [castxml-windows.zip](https://data.kitware.com/api/v1/file/5e8b73e82660cbefba9440a2/download)

解压后将bin目录添加到环境变量PATH中



#### 安装 make
https://gnuwin32.sourceforge.net/packages/make.htm

将 C:\Program Files (x86)\GnuWin32\bin 添加到环境变量PATH中


### 最后安装OMPL


```
conda activate omnigibson
mkdir -p build/Release
cd build/Release

$env:PYTHON_PATH = "C:\Storage\code_external\anaconda3\envs\omnigibson"
$env:BOOST_ROOT = "C:\Storage\code_external\boost_1_82_0"
$env:BOOST_NUMPY_ROOT = "C:\Program Files (x86)\boost.numpy"

cmake ../.. -DPYTHON_EXEC="$env:PYTHON_PATH\python.exe" -DPYTHON_INCLUDE_DIRS="$env:PYTHON_PATH\include\python3.10" -DPYTHON_LIBRARIES="$env:PYTHON_PATH\libs\python3.10.lib" -DBOOST_ROOT="$env:BOOST_ROOT" -DBOOST_LIBRARYDIR="$env:BOOST_ROOT\lib64-msvc-14.3" -DBoost_DIR="$env:BOOST_ROOT\lib64-msvc-14.3\cmake\Boost-1.82.0" -DBOOST_NUMPY_LIBRARY="$env:BOOST_NUMPY_ROOT\lib"

-DBOOST_NUMPY_INCLUDE_DIR="$env:BOOST_NUMPY_ROOT\libs\numpy\src" 

#注意这里一定要把路径改成自己的omnigibson虚拟环境下的python可执行文件路径
make -j 32 update_bindings
make -j 32 
```




## 安装 Fast-Downward

Fast-Downward是一个经典规划求解器，可以用来求解PDDL问题。

```shell
cd btgym/planning/downward
./build.py
```



## vscode 有波浪线导包错误
选择正确的 python 解释器 'ctrl+shift+p'
在 vscode 中对 Pylance 重新禁用再启用

如果 omnigison 库有波浪线，在settings.json中添加，XXX为本地安装的Omnigibson库路径
```json
"python.analysis.extraPaths": [
    "XXX/Omnigibson"
]
```




# 开发规范

## 自己写的测试文件放在 tests 文件夹下

tests 中的文件不会被github同步，所以一些不稳定、不需要共享的临时测试代码可以放在里面。