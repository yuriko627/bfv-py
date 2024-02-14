import unittest
import numpy as np
from bfv.polynomial import (
    PolynomialRing,
    Polynomial,
    poly_mul_naive,
)
import random
from mpmath import *

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
        n = 1024
        q = 7
        Rq = PolynomialRing(n, q)
        aq1 = Rq.sample_polynomial()
        aq2 = Rq.sample_polynomial()

        # Ensure that the coefficients of the polynomial are within the range [-(q-1)/2, (q-1)/2]
        lower_bound = -(q - 1) / 2 # inclusive
        upper_bound = (q - 1) / 2 # inclusive
        for coeff in aq1.coefficients:
            self.assertTrue(coeff >= lower_bound and coeff <= upper_bound)
        for coeff in aq2.coefficients:
            self.assertTrue(coeff >= lower_bound and coeff <= upper_bound)

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
        coefficients = [7, 1, 0]
        a = Polynomial(coefficients)

        # Reduce the coefficients by the modulus of the polynomial ring
        a.reduce_coefficients_by_modulus(Rq.modulus)

        self.assertTrue(np.array_equal(a.coefficients, [0, 1, 0]))

    def test_add_poly(self):
        coefficients_1 = [3, 3, 4, 4, 4]
        coefficients_2 = [3, 2, 0, 1]
        aq1 = Polynomial(coefficients_1)
        aq2 = Polynomial(coefficients_2)
        result = aq1 + aq2
        assert result.coefficients == [3, 6, 6, 4, 5]

    def test_add_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients_1 = [3, 3, 4, 4, 4]
        r = Polynomial(coefficients_1)

        r.reduce_in_ring(Rq)

        assert r.coefficients == [3, -3, -3, 1]

        coefficients_2 = [3, 3, 2, 0, 1]
        p = Polynomial(coefficients_2)

        # r + p
        result = r + p

        result.reduce_in_ring(Rq)

        assert result.coefficients == [-1, -1, -3, -1]

    def test_mul_poly_in_ring_Rq(self):
        n = 1024
        q = random.getrandbits(60)
        Rq = PolynomialRing(n, q)
        coeffs1 = [random.getrandbits(60) for _ in range(n)]
        coeffs2 = [random.getrandbits(60) for _ in range(n)]

        aq1 = Polynomial(coeffs1)
        aq2 = Polynomial(coeffs2)

        # aq1 * aq2
        result = aq1 * aq2

        result.reduce_in_ring(Rq)

        # perform the multiplication naively
        product_naive = poly_mul_naive(coeffs1, coeffs2)

        poly_product_naive = Polynomial(product_naive)
        poly_product_naive.reduce_in_ring(Rq)

        assert result.coefficients == poly_product_naive.coefficients


    def test_scalar_mul_poly_in_ring_Rq(self):
        n = 1024
        q = random.getrandbits(60)
        Rq = PolynomialRing(n, q)
        poly1 = Polynomial([random.getrandbits(60) for _ in range(n)])
        scalar = 3

        result = poly1.scalar_mul(scalar)
        result.reduce_in_ring(Rq)

        # perform the scalar multiplication naively
        product_naive = [scalar * coeff for coeff in poly1.coefficients]

        poly_product_naive = Polynomial(product_naive)
        poly_product_naive.reduce_in_ring(Rq)

        for i in range(len(result.coefficients)):
            assert result.coefficients[i] == poly_product_naive.coefficients[i]


    def test_poly_eval(self):
        # random sample 1024 coefficients in the range 0, 1152921504606584833
        coefficients_1 = []
        for _ in range(1024):
            coefficients_1.append(random.randint(0, 1152921504606584833))
        
        coefficients_2 = []
        for _ in range(1024):
            coefficients_2.append(random.randint(0, 1152921504606584833))

        coefficients_3 = []
        for _ in range(1024):
            coefficients_3.append(random.randint(0, 1152921504606584833))

        aq1 = Polynomial(coefficients_1)
        aq2 = Polynomial(coefficients_2)
        aq3 = Polynomial(coefficients_3)

        # aq1 + aq2 * aq3
        result = aq1 + aq2 * aq3

        # evaluate the polynomial at a random x
        x = random.randint(0, 1152921504606584833)
        result = result.evaluate(x)

        aq1 = aq1.evaluate(x)
        aq2 = aq2.evaluate(x)
        aq3 = aq3.evaluate(x)

        # check if the result is equal to the sum of the two polynomials evaluated at x
        assert result == aq1 + aq2 * aq3

        
class TestCenteredRemainder(unittest.TestCase):

    def test_into_centered_coefficients(self):

        mod = 7
        val = random.randint(-100, 100)
        res = Polynomial([val]).into_centered_coefficients(mod).coefficients[0]

        # assert that the result is in the range [-(mod-1)/2, (mod-1)/2]
        assert res >= -(mod - 1) / 2 and res <= (mod - 1) / 2

    def test_into_standard_form(self):

        mod = 7
        val = random.randint(-100, 100)
        res = Polynomial([val]).into_standard_form(mod).coefficients[0]

        # assert that the result is in the range [0, mod-1]
        assert res >= 0 and res <= mod - 1
        assert res == val % mod