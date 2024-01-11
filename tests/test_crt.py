import unittest

from bfv.crt import Q, CRTInteger, CRTPolynomial
import random as rand

from bfv.polynomial import Polynomial, PolynomialRing, poly_add, poly_mul


class TestQ(unittest.TestCase):
    def test_init_q_valid(self):
        qis = [2, 3, 5]
        q = Q(qis)
        self.assertEqual(q.qis, qis)

    def test_init_q_valid_2(self):
        qis = [
            1152921504606584833,
            1152921504598720513,
            1152921504597016577,
            1152921504595968001,
            1152921504595640321,
            1152921504593412097,
            1152921504592822273,
            1152921504592429057,
            1152921504589938689,
            1152921504586530817,
            1152921504585547777,
            1152921504583647233,
            1152921504581877761,
            1152921504581419009,
            1152921504580894721,
        ]
        q = Q(qis)
        self.assertEqual(q.qis, qis)

    def test_init_q_invalid_no_coprime(self):
        qis = [2, 3, 9]
        with self.assertRaisesRegex(AssertionError, "qis are not pairwise coprime"):
            Q(qis)

    def test_init_q_invalid_too_large(self):
        qis = [2, 3, 2**61]
        with self.assertRaisesRegex(AssertionError, "qi is too large"):
            Q(qis)


class TestCRTInteger(unittest.TestCase):
    def test_from_crt_components_valid(
        self,
    ):  # from tutorial https://www.youtube.com/watch?v=zIFehsBHB8o
        qis = [5, 7, 8]
        q = Q(qis)
        xis = [3, 1, 6]
        crt_integer = CRTInteger.from_crt_components(q, xis)
        self.assertEqual(crt_integer.recover(), 78)

    def test_from_integer_valid(self):
        qis = [5, 7, 8]
        q = Q(qis)
        x = rand.randint(0, q.q - 1)
        crt_integer = CRTInteger.from_integer(q, x)
        self.assertEqual(crt_integer.xis, [x % qi for qi in q.qis])
        self.assertEqual(crt_integer.recover(), x)


class TestPolynomialWithCRT(unittest.TestCase):
    def setUp(self):
        self.qis = [
            1152921504606584833,
            1152921504598720513,
            1152921504597016577,
            1152921504595968001,
            1152921504595640321,
            1152921504593412097,
            1152921504592822273,
            1152921504592429057,
            1152921504589938689,
            1152921504586530817,
            1152921504585547777,
            1152921504583647233,
            1152921504581877761,
            1152921504581419009,
            1152921504580894721,
        ]
        self.q = Q(self.qis)
        self.n = 1024
        self.rq = PolynomialRing(self.n, self.q.q)

    def test_valid_polynomial_in_crt_representation(self):
        a = self.rq.sample_polynomial()

        # Reduce polynomial `a` coefficients (with coefficients in Q) to its CRT representations
        rqi_polynomials = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(a, self.q)

        # Recover polynomial `a` from its CRT representations
        a_recovered = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            rqi_polynomials, self.q
        )

        assert a.coefficients == a_recovered.coefficients

    def test_valid_poly_addition_in_crt_representation(self):
        a = self.rq.sample_polynomial()
        b = self.rq.sample_polynomial()
        c = poly_add(a.coefficients, b.coefficients)
        c = Polynomial(c, self.rq)

        # Reduce a coefficients to its CRT representations
        a_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(a, self.q)

        # Reduce b coefficients to its CRT representations
        b_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(b, self.q)

        # Perform a + b in Rqis
        c_rqis = []
        for i in range(len(self.q.qis)):
            c_rqi = poly_add(a_rqis[i].coefficients, b_rqis[i].coefficients)
            rqi = PolynomialRing(self.n, self.q.qis[i])
            c_rqis.append(Polynomial(c_rqi, rqi))

        # Recover c from its CRT representations
        c_recovered = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c_rqis, self.q
        )

        # ensure that a + b = c
        assert c.coefficients == c_recovered.coefficients

    def test_valid_scalar_poly_mul_in_crt_representation(self):
        a = self.rq.sample_polynomial()
        b = self.rq.sample_polynomial()
        c = poly_mul(a.coefficients, b.coefficients)
        c = Polynomial(c, self.rq)

        # Reduce a coefficients and b coefficients to its CRT representations
        a_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(a, self.q)
        b_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(b, self.q)

        # Perform a * b in Rqis
        c_rqis = []
        for i in range(len(self.q.qis)):
            c_rqi = poly_mul(a_rqis[i].coefficients, b_rqis[i].coefficients)
            rqi = PolynomialRing(self.n, self.q.qis[i])
            c_rqis.append(Polynomial(c_rqi, rqi))

        # Recover c from its CRT representations
        c_recovered = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c_rqis, self.q
        )

        # ensure that a * b = c
        assert c.coefficients == c_recovered.coefficients


if __name__ == "__main__":
    unittest.main()
