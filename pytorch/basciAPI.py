import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)


x = torch.tensor([5.5, 3])
print(x)


y = torch.rand(5, 3)
print(x + y)


print(torch.add(x, y))


result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)


# 索引
# 我们还可以使用类似NumPy的索引操作来访问Tensor的一部分，需要注意的是：索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
y = x[0, :]
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了


y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())

# 使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor

x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)


# 另外一个常用的函数就是item(), 它可以将一个标量Tensor转换成一个Python number：
x = torch.randn(1)
print(x)
print(x.item())


