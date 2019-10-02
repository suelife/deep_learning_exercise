class Man:

    def __init__(self, name):
        self.name = name
        print('Initilized')

    def hello(self):
        print('Hello {} !'.format(self.name))

    def goodbye(self):
        print('Good-bye {} !'.format(self.name))


m = Man('jack')
m.hello()
m.goodbye()