# BTGym
A Platform for Behavior Tree Designing and Planning

# 安装

## 安装BDDL
BDDL是用来解析Behavior-1K任务的库

直接通过github仓库安装即可
```python
git clone https://github.com/StanfordVL/bddl.git
```

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
sudo apt-get install libboost-all-dev
sudo apt-get install cmake
sudo apt-get update sudo apt-get install libeigen3-dev
```


#### 安装OMPL
```
conda activate omnigibson
mkdir -p build/Release
cd build/Release
cmake ../.. -DPYTHON_EXEC=/home/cxl/softwares/anaconda3/envs/omnigibson/bin/python #注意这里一定要把路径改成自己的omnigibson虚拟环境下的python可执行文件路径
make -j 32 update_bindings
make -j 32 # replace
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


