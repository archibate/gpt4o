def fib(x):
    if x <= 1:
        return x
    else:
        return fib(x-1) + fib(x-2)

def test_fib():
    assert fib(0) == 0
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(5) == 5
    assert fib(10) == 55
