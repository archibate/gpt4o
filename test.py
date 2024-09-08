import unittest

def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)

class TestFibFunction(unittest.TestCase):
    def test_fib(self):
        self.assertEqual(fib(0), 0)
        self.assertEqual(fib(1), 1)
        self.assertEqual(fib(2), 1)
        self.assertEqual(fib(3), 2)
        self.assertEqual(fib(4), 3)
        self.assertEqual(fib(5), 5)

if __name__ == '__main__':
    unittest.main()
