class MyClass:
    def __init__(self):
        setattr(self, "pro", 2)


if __name__ == '__main__':
    cls = MyClass()
    print(cls.pro)
