from pyplusplus import module_builder
import os

# 创建一个简单的C++测试文件
cpp_code = """
class TestClass {
public:
    int add(int a, int b) { return a + b; }
};
"""

with open("test.hpp", "w") as f:
    f.write(cpp_code)

# 测试Py++
try:
    mb = module_builder.module_builder_t(
        files=["test.hpp"],
        xml_generator_path=r"castxml",  # 确保castxml在PATH中
        compiler="msvc14"  # 使用Visual Studio编译器
    )
    print("Py++ 安装成功！")
except Exception as e:
    print(f"错误: {e}")

# 清理测试文件
os.remove("test.hpp")