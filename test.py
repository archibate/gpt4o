def fib(x):
    if x <= 1:
        return x
    else:
        return fib(x-1) + fib(x-2)

def test_fib():
    assert fib(0) == 0
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(3) == 2
    assert fib(4) == 3
    assert fib(5) == 5
    assert fib(6) == 8
    assert fib(7) == 13
    assert fib(8) == 21
    assert fib(9) == 34
    assert fib(10) == 55
