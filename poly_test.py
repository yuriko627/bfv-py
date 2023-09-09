import unittest
import numpy as np
from polynomial import PolynomialRing, Polynomial, custom_modulo

class TestPolynomialRing(unittest.TestCase):
	def test_init_with_n(self):
		n = 4
		R = PolynomialRing(n)
		quotient = np.array([1, 0, 0, 0, 1])
		self.assertTrue(np.array_equal(R.denominator, quotient))
		self.assertEqual(R.n, n)

	def test_init_with_n_and_q(self):
		n = 4
		q = 7
		Rq = PolynomialRing(n, q)
		quotient = np.array([1, 0, 0, 0, 1])
		self.assertTrue(np.array_equal(Rq.denominator, quotient))
		self.assertEqual(Rq.Q, q)
		self.assertEqual(Rq.n, n)

	def test_init_with_n_and_invalid_q(self):
		n = 4
		q = 1
		with self.assertRaisesRegex(AssertionError, "modulus must be > 1"): PolynomialRing(n, q)

	def test_sample_poly_from_r_error(self):
		n = 4
		R = PolynomialRing(n)
		with self.assertRaisesRegex(AssertionError, "The modulus Q must be set to sample a polynomial from R_Q"): R.sample_polynomial()

	def test_sample_poly_from_rq(self):
		n = 4
		q = 7
		Rq = PolynomialRing(n, q)
		aq1 = Rq.sample_polynomial()
		aq2 = Rq.sample_polynomial()

		# Ensure that the coefficients of the polynomial are within Z_q = (-q/2, q/2]
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

class TestPolynomialInRingR(unittest.TestCase):

    def test_init_poly_in_ring_R(self):
        n = 4
        R = PolynomialRing(n)
        coefficients = [6, 6, 6, 4, 5]
        # a is the polynomial in R reduced by the quotient polynomial
        a = Polynomial(coefficients, R)
        self.assertTrue(np.array_equal(a.coefficients, [6, 6, 4, -1]))
        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = np.polydiv(a.coefficients, a.ring.denominator)
        self.assertTrue(np.array_equal(remainder, a.coefficients))

    def test_add_poly_in_ring_R(self):
        n = 4
        R = PolynomialRing(n)
        coefficients_1 = [3, 3, 4, 4, 4]
        coefficients_2 = [3, 3, 2, 0, 1]

        a1 = Polynomial(coefficients_1, R)
        a2 = Polynomial(coefficients_2, R)

        # a1 + a2
        result = np.polyadd(a1.coefficients, a2.coefficients)

        # The addition is happening in the ring R.
        result = Polynomial(result, R)

        # The resulting poly is 6, 6, 6, 4, 5. After reduction, the polynomial is 6, 6, 4, -1.
        self.assertTrue(np.array_equal(result.coefficients, [6, 6, 4, -1]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = np.polydiv(result.coefficients, result.ring.denominator)
        self.assertTrue(np.array_equal(remainder, result.coefficients))

        # The degree of the result poly should be the max of the two degrees
        self.assertEqual(len(result.coefficients), max(len(a1.coefficients), len(a2.coefficients)))

    def test_mul_poly_in_ring_R(self):
        n = 4
        R = PolynomialRing(n)
        coefficients_1 = [3, 0, 4]
        coefficients_2 = [2, 0, 1]

        a1 = Polynomial(coefficients_1, R)
        a2 = Polynomial(coefficients_2, R)

        result = np.polymul(a1.coefficients, a2.coefficients)

        # The multiplication is happening in the ring R.
        result = Polynomial(result, R)

        self.assertTrue(np.array_equal(result.coefficients, [11, 0, -2]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = np.polydiv(result.coefficients, result.ring.denominator)
        self.assertTrue(np.array_equal(remainder, result.coefficients))

        # The degree of the result poly should less than the degree of fx
        self.assertTrue(len(result.coefficients) < len(result.ring.denominator))

    def test_scalar_mul_poly_in_ring_R(self):
        n = 4
        R = PolynomialRing(n)
        coefficients = [4, 3, 0, 4]
        a = Polynomial(coefficients, R)

        # 2 * a. The resulting poly is 8, 6, 0, 8. Reduction does not change the coefficients.
        result = np.polymul(2, a.coefficients)

        # The multiplication is happening in the ring R.
        result = Polynomial(result, R)

        self.assertTrue(np.array_equal(result.coefficients, [8, 6, 0, 8]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = np.polydiv(result.coefficients, result.ring.denominator)
        self.assertTrue(np.array_equal(remainder, result.coefficients))

class TestPolynomialInRingRq(unittest.TestCase):

    def test_init_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients = [3, 1, 0]
        # fetch the Z_q set which is (-q/2, q/2]
        assert Rq.Z_Q == [-3, -2, -1, 0, 1, 2, 3]
        # aq is the polynomial in Rq reduced by the quotient polynomial and by the modulus q.
        aq = Polynomial(coefficients, Rq)
        self.assertTrue(np.array_equal(aq.coefficients, [3, 1, 0]))
        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = np.polydiv(aq.coefficients, aq.ring.denominator)
        self.assertTrue(np.array_equal(remainder, aq.coefficients))
        # All coefficients should be in Z_q
        for coeff in aq.coefficients:
            self.assertTrue(coeff in Rq.Z_Q)


    def test_add_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients_1 = [3, 3, 4, 4, 4] # r(x)

        # 1st reduction
        _, remainder = np.polydiv(coefficients_1, Rq.denominator) # r(x)/f(x). where f(x)= x^n + 1

        # 2nd reduction
        for i in range(len(remainder)):
            remainder[i] = custom_modulo(remainder[i], Rq.Q)

        aq1 = Polynomial(coefficients_1, Rq)

        # check that coefficents of aq1 and remainder are the same
        self.assertTrue(np.array_equal(aq1.coefficients, remainder))

        coefficients_2 = [3, 3, 2, 0, 1]

        aq2 = Polynomial(coefficients_2, Rq)

        # aq1 + aq2
        result = np.polyadd(aq1.coefficients, aq2.coefficients)

        # The addition is happening in the ring Rq.
        result = Polynomial(result, Rq)

        # The resulting poly is 6, 6, 6, 4, 5. After first reduction, the polynomial is 6, 6, 4, -1.  After second reduction (modulo q), the polynomial is -1. -1, -3, -1.
        self.assertTrue(np.array_equal(result.coefficients, [-1, -1, -3, -1]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = np.polydiv(result.coefficients, result.ring.denominator)
        self.assertTrue(np.array_equal(remainder, result.coefficients))

        # The degree of the result poly should be the max of the two degrees
        self.assertEqual(len(result.coefficients), max(len(aq1.coefficients), len(aq2.coefficients)))

    def test_mul_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients_1 = [3, 0, 4]
        coefficients_2 = [2, 0, 1]

        aq1 = Polynomial(coefficients_1, Rq)
        aq2 = Polynomial(coefficients_2, Rq)

        # aq1 * aq2. After reduction, the polynomial is . After second reduction (modulo q), the polynomial is -3, 0, -2.
        result = np.polymul(aq1.coefficients, aq2.coefficients)

        # The multiplication is happening in the ring Rq.
        result = Polynomial(result, Rq)

        self.assertTrue(np.array_equal(result.coefficients, [-3, 0, -2]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = np.polydiv(result.coefficients, result.ring.denominator)
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
        result = np.polymul(2, aq.coefficients)

        # The multiplication is happening in the ring Rq.
        result = Polynomial(result, Rq)

        self.assertTrue(np.array_equal(result.coefficients, [1, -1, 0, 1]))

        # After reduction, the polynomial should not be multiple of the quotient
        _, remainder = np.polydiv(result.coefficients, result.ring.denominator)
        self.assertTrue(np.array_equal(remainder, result.coefficients))


class TestCustomModulo(unittest.TestCase):

    def test_positive_values(self):
        self.assertEqual(custom_modulo(7, 10), -3)  # 7 % 10. Lies in the range (-5, 5]
        self.assertEqual(custom_modulo(15, 10), 5) # 15 % 10 = 5, which is <= 5
        self.assertEqual(custom_modulo(17, 10), -3) # 17 % 10 = 7, which is > 5. So, 7 - 10 = -3

    def test_negative_values(self):
        self.assertEqual(custom_modulo(-7, 10), 3) # Lies in the range (-5, 5]
        self.assertEqual(custom_modulo(-15, 10), 5) # -15 % 10 = 5 (in Python, % returns non-negative), which is <= 5
        self.assertEqual(custom_modulo(-17, 10), 3) # -17 % 10 = 3, which is <= 5

    def test_boundary_values(self):
        q = 7
        self.assertEqual(custom_modulo(-q//2 + 1, q), -q//2 + 1) # The smallest positive number in the range
        self.assertEqual(custom_modulo(q//2, q), q//2) # The largest number in the range

    def test_zero(self):
        self.assertEqual(custom_modulo(0, 10), 0) # 0 lies in the range (-5, 5]

if __name__ == "__main__":
    unittest.main()