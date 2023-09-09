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

	# def test_sample_poly_from_rq(self):
	# 	n = 4
	# 	q = 7
	# 	Rq = PolynomialRing(n, q)
	# 	aq1 = Rq.sample_polynomial()
	# 	aq2 = Rq.sample_polynomial()

	# 	# Ensure that the coefficients of the polynomial are within Z_q = (-q/2, q/2]
	# 	for coeff in aq1.coefficients:
	# 		self.assertTrue(coeff >= -q // 2 and coeff <= q // 2)
	# 	for coeff in aq2.coefficients:
	# 		self.assertTrue(coeff >= -q // 2 and coeff <= q // 2)

    #     # Ensure that the degree of the sampled poly is equal or less than d (it might be less if the leading coefficient sampled is 0)
	# 	count1 = 0
	# 	for coeff in aq1.coefficients:
	# 		count1 += 1

	# 	count2 = 0
	# 	for coeff in aq2.coefficients:
	# 		count2 += 1

	# 	self.assertTrue(count1 <= Rq.n)
	# 	self.assertTrue(count2 <= Rq.n)

if __name__ == "__main__":
    unittest.main()