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


### Ubuntu 安装依赖

1. 直接用apt安装依赖
```shell
sudo apt-get install libboost-all-dev
sudo apt-get install cmake
sudo apt-get update sudo apt-get install libeigen3-dev
```


### Windows 安装依赖

#### Boost

参考[博客](https://blog.csdn.net/qq_38967414/article/details/129347708?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-129347708-blog-141728930.235%5Ev43%5Epc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-129347708-blog-141728930.235%5Ev43%5Epc_blog_bottom_relevance_base3&utm_relevant_index=1)进行安装

1. 在官网下载最新版本，例如：[boost_1_86_0.zip](https://archives.boost.io/release/1.86.0/source/boost_1_86_0.zip)，并解压
2. 计算机中搜索并打开 Developer Command Prompt for VS 2022
3. 安装
```
cd boost_1_86_0
bootstrap.bat
.\b2
```


#### CMake
在官网下载最新版本，例如：[cmake-3.30.5-windows-x86_64.msi](https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-windows-x86_64.msi)

验证
```
cmake --version
```



### 从源码安装OMPL
```
mkdir -p build/Release
cd build/Release
cmake ../.. -DPYTHON_EXEC=/home/cxl/softwares/anaconda3/envs/omnigibson/bin/python #注意这里一定要把路径改成自己的omnigibson虚拟环境下的python可执行文件路径
make -j 32 update_bindings
make -j 32 # replace
```


## 安装 Fast-Downward

Fast-Downward是一个经典规划求解器，可以用来求解PDDL问题。

```shell
cd btp/planning/downward
./build.py
```

