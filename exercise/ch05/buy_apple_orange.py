from layer_naive import MulLayer
from layer_naive import AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# Layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_fruit_layer = AddLayer()
mul_tax_layer = MulLayer()

# Forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
price = add_fruit_layer.forward(apple_price, orange_price)
total_price = mul_tax_layer.forward(price, tax)

# Backward
dtotal_price = 1
dprice, dtax = mul_tax_layer.backward(dtotal_price)
dapple_price, dorange_price = add_fruit_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print('apple_price: ', apple_price)
print('orange_price:', orange_price)
print('price:', price)
print('total_price:', int(total_price))
print('='*50)
print('dprice: ', dprice)
print('dtax: ', dtax)
print('dapple_price: ', dapple_price)
print('dorange_price: ', dorange_price)
print('dapple: ', dapple)
print('dapple_num: ', int(dapple_num))
print('dorange: ', int(dorange))
print('dorange_num: ', int(dorange_num))
