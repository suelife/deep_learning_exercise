def add(x, y):
    return lambda z: x+y+z

x = 10
y = 2

a = add(x, y)
b = a(-30)

print(b)


print('hello')