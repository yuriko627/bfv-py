import unittest

from bfv.crt import CRTModuli, CRTInteger, CRTPolynomial
from bfv.polynomial import PolynomialRing, Polynomial, poly_mul_naive
from bfv.utils import find_odd_pairwise_coprimes
import random
from mpmath import *

class TestQ(unittest.TestCase):
    def test_init_q_valid(self):

        start = random.getrandbits(60)
        qis = find_odd_pairwise_coprimes(start, 15)
        crt_moduli = CRTModuli(qis)
        self.assertEqual(crt_moduli.qis, qis)

    def test_init_q_invalid_no_coprime(self):
        qis = [2, 3, 9]
        with self.assertRaisesRegex(AssertionError, "qis are not pairwise coprime"):
            CRTModuli(qis)

    def test_init_q_invalid_too_large(self):
        qis = [2, 3, 2**61]
        with self.assertRaisesRegex(AssertionError, "qi is too large"):
            CRTModuli(qis)


class TestCRTInteger(unittest.TestCase):
    def test_from_crt_components_valid(
        self,
    ):  # from tutorial https://www.youtube.com/watch?v=zIFehsBHB8o
        qis = [5, 7, 8]
        q = CRTModuli(qis)
        xis = [3, 1, 6]
        crt_integer = CRTInteger.from_crt_components(q, xis)
        self.assertEqual(crt_integer.recover(), 78)

    def test_from_integer_valid(self):
        qis = [5, 7, 8]
        q = CRTModuli(qis)
        x = random.randint(0, q.q - 1)
        crt_integer = CRTInteger.from_integer(q, x)
        self.assertEqual(crt_integer.xis, [x % qi for qi in q.qis])
        self.assertEqual(crt_integer.recover(), x)


class TestPolynomialWithCRTAndRandomCoprimes(unittest.TestCase):
    def setUp(self):
        start = random.getrandbits(60)
        self.qis = find_odd_pairwise_coprimes(start, 15)
        self.crt_moduli = CRTModuli(self.qis)
        self.n = 1024
        self.rq = PolynomialRing(self.n, self.crt_moduli.q)

    def test_valid_polynomial_in_crt_representation(self):
        # Sample polynomial `a` from the ring R_q
        a = self.rq.sample_polynomial()

        # Reduce polynomial `a` to its CRT representations - namely to polynomials in R_qi
        rqi_polynomials = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            a, self.n, self.crt_moduli
        )

        # Recover polynomial `a` from its CRT representations - namely to a polynomial in R_q
        a_recovered = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            rqi_polynomials, self.n, self.crt_moduli
        )

        assert a.coefficients == a_recovered.coefficients

    def test_valid_poly_addition_in_crt_representation(self):
        a = self.rq.sample_polynomial()
        b = self.rq.sample_polynomial()
        c = a + b
        c.reduce_coefficients_by_modulus(self.rq.modulus)

        # Reduce a coefficients to its CRT representations -> `a_rqis` are polynomials in R_qi
        a_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            a, self.n, self.crt_moduli
        )

        # Reduce b coefficients to its CRT representations -> `b_rqis` are polynomials in R_qi
        b_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            b, self.n, self.crt_moduli
        )

        # Perform a + b in crt representation
        c_rqis = []
        for i in range(len(self.crt_moduli.qis)):
            c_rqi = a_rqis[i] + b_rqis[i]
            c_rqi.reduce_coefficients_by_modulus(self.rq.modulus)
            c_rqis.append(c_rqi)

        # Recover c from its CRT representations
        c_recovered = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c_rqis, self.n, self.crt_moduli
        )

        # ensure that a + b = c
        assert c.coefficients == c_recovered.coefficients

    def test_valid_poly_mul_in_crt_representation(self):
        a = self.rq.sample_polynomial()
        b = self.rq.sample_polynomial()
        c = poly_mul_naive(a.coefficients, b.coefficients)
        c = Polynomial(c)
        c.reduce_in_ring(self.rq)

        # Reduce a coefficients and b coefficients to its CRT representations
        a_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            a, self.n, self.crt_moduli
        )
        b_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            b, self.n, self.crt_moduli
        )

        # Perform a * b in crt representation
        c_rqis = []
        for i in range(len(self.crt_moduli.qis)):
            c_rqi = a_rqis[i] * b_rqis[i]
            rqi = PolynomialRing(self.n, self.crt_moduli.qis[i])
            c_rqi.reduce_in_ring(rqi)
            c_rqis.append(c_rqi)
        # Recover c from its CRT representations
        c_recovered = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c_rqis, self.n, self.crt_moduli
        )

        # ensure that c_recovered = c
        assert c.coefficients == c_recovered.coefficients