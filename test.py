def fib(x):
    if x <= 1:
        return x
    elif x == 2:
        return 1
    else:
        return fib(x-1) + fib(x-2)

def test_fib():
    outputs = [0, 1, 1, 2, 3, 5]
    for i, expected in enumerate(outputs):
        assert fib(i) == expected
