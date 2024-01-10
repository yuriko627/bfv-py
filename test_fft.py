import unittest
from fft import Rou, FFT
import math


class TestRootsOfUnity(unittest.TestCase):
    def test_init_rou(self):
        rou = Rou(4)
        self.assertEqual(rou.n, 4)
        roots = rou.get_roots()
        self.assertAlmostEqual(roots[0], complex(1, 0))
        self.assertAlmostEqual(roots[1], complex(0, 1))
        self.assertAlmostEqual(roots[2], complex(-1, 0))
        self.assertAlmostEqual(roots[3], complex(0, -1))

        rou = Rou(3)
        self.assertEqual(rou.n, 3)
        roots = rou.get_roots()

        self.assertAlmostEqual(roots[0], complex(1, 0))
        self.assertAlmostEqual(roots[1], complex(-0.5, math.sqrt(3) / 2))
        self.assertAlmostEqual(roots[2], complex(-0.5, -math.sqrt(3) / 2))

        base = rou.get_base()
        self.assertAlmostEqual(base, complex(-0.5, math.sqrt(3) / 2))

    def test_fft(self):
        fft = FFT()
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        y = fft.recursive_fft(a)


if __name__ == "__main__":
    unittest.main()
