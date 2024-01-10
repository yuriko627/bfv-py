import math
from typing import List
import matplotlib.pyplot as plt


class Rou:
    def __init__(self, n):
        self.n = n
        self.angle = 360 / n

    def get_roots(self):
        roots = []
        for k in range(self.n):
            theta = self.angle * k
            root = complex(math.cos(math.radians(theta)), math.sin(math.radians(theta)))
            roots.append(root)
        return roots

    def get_base(self):
        return complex(
            math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        )

    def plot_roots(self):
        roots = self.get_roots()
        plt.figure(figsize=(6, 6))
        plt.axhline(y=0, color="k")
        plt.axvline(x=0, color="k")
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.grid(True, which="both")

        # Draw the unit circle
        circle = plt.Circle((0, 0), 1, color="blue", fill=False)
        plt.gca().add_artist(circle)

        for root in roots:
            plt.plot(root.real, root.imag, "ro")  # 'ro' means red circle markers

        plt.title(f"{self.n}-th Roots of Unity")
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()


class FFT:
    def recursive_fft(self, a: List[int]):
        """
        Performs a recursive FFT on a polynomial `a`.
        """
        n = len(a)
        if n == 1:
            return a
        rou = Rou(n)
        rou_base = rou.get_base()
        rou_acc = complex(1, 0)
        a_even = a[::2]
        a_odd = a[1::2]
        y_even = self.recursive_fft(a_even)
        y_odd = self.recursive_fft(a_odd)
        y = [0] * n
        for k in range(n // 2):
            y[k] = y_even[k] + rou_acc * y_odd[k]
            y[k + n // 2] = y_even[k] - rou_acc * y_odd[k]
            rou_acc = rou_acc * rou_base
        print("for loop done for n", n - 1)
        return y
