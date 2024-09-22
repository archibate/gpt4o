from typing import Callable, Generic, Optional, TypeVar, List
from contextlib import contextmanager
import unittest

K = TypeVar('K')
V = TypeVar('V')

class LruCache(Generic[K, V]):
    def __init__(self):
        self.cache: dict[K, V] = {}
        self.lru: List[K] = []
        # TODO: implement lru deletion

    @contextmanager
    def usage_guard(self):
        yield

    def find_entry(self, key: K) -> Optional[V]:
        return self.cache.get(key)

    def add_entry(self, key: K, val: V):
        self.cache[key] = val

    def delete_entry(self, key: K):
        self.cache.pop(key, None)

    def cached_calc(self, calc: Callable[[K], V], input: K):
        with self.usage_guard():
            output = self.find_entry(input)
            if output is None:
                output = calc(input)
                self.add_entry(input, output)
        return output

    def batched_cached_calc(self,
                            calc: Callable[[List[K]], List[V]],
                            inputs: List[K]) -> List[V]:
        missed_inputs: List[K] = []
        missed_indices: List[int] = []
        outputs: List[Optional[V]] = []

        with self.usage_guard():
            for i, input in enumerate(inputs):
                output = self.find_entry(input)
                if output is None:
                    missed_inputs.append(input)
                    missed_indices.append(i)
                    outputs.append(None)
                else:
                    outputs.append(output)

            if missed_inputs:
                missed_outputs = calc(missed_inputs)
                assert isinstance(missed_outputs, List)
                for i, output in zip(missed_indices, missed_outputs):
                    outputs[i] = output
                    self.add_entry(inputs[i], output)

        assert all(x is not None for x in outputs)
        return outputs  # type: ignore


class TestLruCache(unittest.TestCase):
    def setUp(self):
        self.cache = LruCache()

    def test_add_entry(self):
        self.cache.add_entry(1, 'a')
        self.assertEqual(self.cache.find_entry(1), 'a')

    def test_delete_entry(self):
        self.cache.add_entry(1, 'a')
        self.cache.delete_entry(1)
        self.assertEqual(self.cache.find_entry(1), None)

    def test_cached_calc(self):
        def calc(x: int) -> int:
            return x + 1

        self.assertEqual(self.cache.cached_calc(calc, 1), 2)
        self.assertEqual(self.cache.cached_calc(calc, 2), 3)
        self.assertEqual(self.cache.cached_calc(calc, 3), 4)
        self.assertEqual(self.cache.cached_calc(calc, 1), 2)

    def test_batched_cached_calc(self):
        def calc(x: List[int]) -> List[int]:
            return [i + 1 for i in x]

        self.assertEqual(self.cache.batched_cached_calc(calc, [1, 2, 3]), [2, 3, 4])
        self.assertEqual(self.cache.batched_cached_calc(calc, [0, 2, 3, 6]), [1, 3, 4, 7])

if __name__ == '__main__':
    unittest.main()
