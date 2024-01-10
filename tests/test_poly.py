import unittest
import numpy as np
from bfv.polynomial import (
    PolynomialRing,
    Polynomial,
    get_centered_remainder,
    poly_add,
    poly_div,
    poly_mul,
)


class TestPolynomialRing(unittest.TestCase):
    def test_init_with_n_and_q(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        quotient = np.array([1, 0, 0, 0, 1])
        self.assertTrue(np.array_equal(Rq.denominator, quotient))
        self.assertEqual(Rq.modulus, q)
        self.assertEqual(Rq.n, n)

    def test_sample_poly_from_rq(self):
        n = 4
        q = 8
        Rq = PolynomialRing(n, q)
        aq1 = Rq.sample_polynomial()
        aq2 = Rq.sample_polynomial()

        # Ensure that the coefficients of the polynomial are within Z_q = [-q/2, q/2)
        for coeff in aq1.coefficients:
            self.assertTrue(coeff >= -q // 2 and coeff <= q // 2)
        for coeff in aq2.coefficients:
            self.assertTrue(coeff >= -q // 2 and coeff <= q // 2)

        # Ensure that the degree of the sampled poly is equal or less than d (it might be less if the leading coefficient sampled is 0)
        count1 = 0
        for coeff in aq1.coefficients:
            count1 += 1

        count2 = 0
        for coeff in aq2.coefficients:
            count2 += 1

        self.assertTrue(count1 <= Rq.n)
        self.assertTrue(count2 <= Rq.n)


class TestPolynomialInRingRq(unittest.TestCase):
    def test_init_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients = [3, 1, 0]
        # aq is the polynomial in Rq reduced by the quotient polynomial and by the modulus q.
        aq = Polynomial(coefficients, Rq)
        self.assertTrue(np.array_equal(aq.coefficients, [0, 3, 1, 0]))
        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = poly_div(aq.coefficients, aq.ring.denominator)
        # pad remainder with zeroes at the beginning to make it n-coefficients long
        remainder = np.concatenate((np.zeros(n - len(remainder)), remainder))
        self.assertTrue(np.array_equal(remainder, aq.coefficients))
        # All coefficients should be in Z_q - (-q/2, q/2]
        Z_q = set()
        for i in range(-q // 2 + 1, q // 2 + 1):
            Z_q.add(i)

        for coeff in aq.coefficients:
            self.assertTrue(coeff in Z_q)

    def test_add_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients_1 = [3, 3, 4, 4, 4]  # r(x)

        # 1st reduction
        _, remainder = poly_div(
            coefficients_1, Rq.denominator
        )  # r(x)/f(x). where f(x)= x^n + 1

        # 2nd reduction
        for i in range(len(remainder)):
            remainder[i] = get_centered_remainder(remainder[i], Rq.modulus)

        aq1 = Polynomial(coefficients_1, Rq)

        # check that coefficents of aq1 and remainder are the same
        self.assertTrue(np.array_equal(aq1.coefficients, remainder))

        coefficients_2 = [3, 3, 2, 0, 1]

        aq2 = Polynomial(coefficients_2, Rq)

        # aq1 + aq2
        result = poly_add(aq1.coefficients, aq2.coefficients)

        # The addition is happening in the ring Rq.
        result = Polynomial(result, Rq)

        # The resulting poly is 6, 6, 6, 4, 5. After first reduction, the polynomial is 6, 6, 4, -1.  After second reduction (modulo q), the polynomial is -1. -1, -3, -1.
        self.assertTrue(np.array_equal(result.coefficients, [-1, -1, -3, -1]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = poly_div(result.coefficients, result.ring.denominator)
        self.assertTrue(np.array_equal(remainder, result.coefficients))

        # The degree of the result poly should be the max of the two degrees
        self.assertEqual(
            len(result.coefficients), max(len(aq1.coefficients), len(aq2.coefficients))
        )

    def test_mul_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients_1 = [3, 0, 4]
        coefficients_2 = [2, 0, 1]

        aq1 = Polynomial(coefficients_1, Rq)
        aq2 = Polynomial(coefficients_2, Rq)

        # aq1 * aq2. After reduction, the polynomial is . After second reduction (modulo q), the polynomial is -3, 0, -2.
        result = poly_mul(aq1.coefficients, aq2.coefficients)

        # The multiplication is happening in the ring Rq.
        result = Polynomial(result, Rq)

        self.assertTrue(np.array_equal(result.coefficients, [0, -3, 0, -2]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = poly_div(result.coefficients, result.ring.denominator)
        # pad remainder with zeroes at the beginning to make it n-coefficients long
        remainder = np.concatenate((np.zeros(n - len(remainder)), remainder))
        self.assertTrue(np.array_equal(remainder, result.coefficients))

        # The degree of the result poly should less than the degree of fx
        self.assertTrue(len(result.coefficients) < len(result.ring.denominator))

    def test_scalar_mul_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients = [4, 3, 0, 4]
        aq = Polynomial(coefficients, Rq)

        # The resulting poly is 8, 6, 0, 8. After modulo q reduction, the polynomial is 1, -1, 0, 1.
        result = poly_mul([2], aq.coefficients)

        # The multiplication is happening in the ring Rq.
        result = Polynomial(result, Rq)

        self.assertTrue(np.array_equal(result.coefficients, [1, -1, 0, 1]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = poly_div(result.coefficients, result.ring.denominator)
        self.assertTrue(np.array_equal(remainder, result.coefficients))


class TestCustomModulo(unittest.TestCase):
    def test_positive_values(self):
        self.assertEqual(
            get_centered_remainder(7, 10), -3
        )  # 7 % 10. Lies in the range (-5, 5]
        self.assertEqual(
            get_centered_remainder(15, 10), 5
        )  # 15 % 10 = 5, which is <= 5
        self.assertEqual(
            get_centered_remainder(17, 10), -3
        )  # 17 % 10 = 7, which is > 5. So, 7 - 10 = -3

    def test_negative_values(self):
        self.assertEqual(get_centered_remainder(-7, 10), 3)  # Lies in the range (-5, 5]
        self.assertEqual(
            get_centered_remainder(-15, 10), 5
        )  # -15 % 10 = 5 (in Python, % returns non-negative), which is <= 5
        self.assertEqual(
            get_centered_remainder(-17, 10), 3
        )  # -17 % 10 = 3, which is <= 5

    def test_boundary_values(self):
        q = 7
        self.assertEqual(
            get_centered_remainder(-q // 2 + 1, q), -q // 2 + 1
        )  # The smallest positive number in the range
        self.assertEqual(
            get_centered_remainder(q // 2, q), q // 2
        )  # The largest number in the range

    def test_zero(self):
        self.assertEqual(
            get_centered_remainder(0, 10), 0
        )  # 0 lies in the range (-5, 5]


if __name__ == "__main__":
    unittest.main()
