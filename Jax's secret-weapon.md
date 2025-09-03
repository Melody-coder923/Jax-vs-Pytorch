###JIT compilation converts your Python functions into super-fast machine code - slow on the first compilation, but blazingly fast afterwards! This is JAX's secret weapon for being so much faster than PyTorch!
Pytorch
```
def add_numbers(a, b):
    return a + b

# 每次调用都要：
# 1. 解释Python代码
# 2. 调用底层数学库
# 3. 返回结果
result = add_numbers(3, 5)  # 每次都重复上述过程
```
JAX
```
@jax.jit  # 这个装饰器是关键！
def add_numbers(a, b):
    return a + b

# 第一次调用：
result = add_numbers(3, 5)  
# JAX做了什么：
# 1. 把Python函数编译成优化的机器码
# 2. 缓存这个机器码
# 3. 执行并返回结果

# 后续调用：
result = add_numbers(4, 6)  # 直接执行缓存的机器码，超快！
```
