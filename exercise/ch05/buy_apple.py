from layer_naive import MulLayer

apple = 100
apple_num = 2
tax = 1.1

# Layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# Forward
apple_price = mul_apple_layer.forward(apple, apple_num) # 200
total_price = mul_tax_layer.forward(apple_price, tax) # 220

# Backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# print('total price:', int(total_price))
# print('dapple:', dapple)
# print('dapple_num:', int(dapple_num))
# print('dtax:', dtax)

print('apple_price: ', apple_price)
print('total_price: ', total_price)
print('='*50)
print('dapple_price', dapple_price)
print('dtax', dtax)
print('dapple', apple)
print('dapple_num', dapple_num)
