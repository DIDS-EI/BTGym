# OBTP
Optimal Behavior Tree Planning

# 安装

## 安装BDDL
BDDL是用来解析Behavior-1K任务的库

直接通过github仓库安装即可
```python
git clone https://github.com/StanfordVL/bddl.git
```

## 安装OMPL
OMPL (the Open Motion Planning Library)，开源运动规划库，由许多基于采样的先进的运动规划算法组成。[官网](https://ompl.kavrakilab.org/download.html)

### Ubuntu
1. 在官网下载[安装脚本](https://ompl.kavrakilab.org/install-ompl-ubuntu.sh)
2. 将第88行
```
cmake ../.. -DPYTHON_EXEC=/usr/bin/python${PYTHONV}
```

后面的python路径改为 omnigibson 虚拟环境中的python路径

3. 按照[官方说明](https://ompl.kavrakilab.org/installation.html)安装依赖
4. 运行
```shell
./install-ompl-ubuntu.sh --python
```

### Windows
从源码安装
https://github.com/ompl/ompl.git

克隆ompl仓库，并按照README安装Boost, CMake, Eigen依赖。

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


#### 最后安装 OMPL

```
cd 
mkdir -p build/Release
cd build/Release
cmake ../..
# next step is optional
make -j 4 update_bindings # if you want Python bindings
make -j 4 # replace "4" with the number of cores on your machine
```
