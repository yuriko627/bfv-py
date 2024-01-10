import math
import unittest

from bfv.discrete_gauss import DiscreteGaussian
from bfv.polynomial import PolynomialRing, Polynomial, get_centered_remainder, poly_add
from bfv.bfv import BFV
from bfv.crt import Q, CRTPolynomial


class TestBFV(unittest.TestCase):
    def setUp(self):
        self.n = 1024
        self.q = 536870909
        self.sigma = 3.2
        self.discrete_gaussian = DiscreteGaussian(self.sigma)
        self.t = 257
        self.delta = int(math.floor(self.q / self.t))
        self.bfv = BFV(self.n, self.q, self.t, self.discrete_gaussian)

    def test_bfv_initialization(self):
        self.assertEqual(
            self.bfv.Rq.denominator, PolynomialRing(self.n, self.q).denominator
        )
        self.assertEqual(self.bfv.Rq.n, PolynomialRing(self.n, self.q).n)
        self.assertEqual(self.bfv.Rq.modulus, PolynomialRing(self.n, self.q).modulus)

        self.assertEqual(
            self.bfv.Rt.denominator, PolynomialRing(self.n, self.t).denominator
        )
        self.assertEqual(self.bfv.Rt.n, PolynomialRing(self.n, self.t).n)
        self.assertEqual(self.bfv.Rt.modulus, PolynomialRing(self.n, self.t).modulus)

        self.assertEqual(self.bfv.distribution, self.discrete_gaussian)

    def test_sample_from_chi_key_distribution(self):
        key = self.bfv.SampleFromTernaryDistribution()
        # Ensure that the degree of the sampled poly is equal or less than d (it might be less if the leading coefficient sampled is 0)
        self.assertTrue(len(key.coefficients) <= self.bfv.n)

        # Ensure that the coefficients of the key are within {-1, 0, 1}
        for coeff in key.coefficients:
            self.assertTrue(coeff == -1 or coeff == 0 or coeff == 1)

    def test_sample_from_chi_error_distribution(self):
        error = self.bfv.SampleFromErrorDistribution()
        # Ensure that the degree of the sample polynomial is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(error.coefficients) <= self.bfv.n)

        # Ensure that the secret key is sampled from the error distribution by checking that each coefficient is within the range of the error distribution
        for coeff in error.coefficients:
            self.assertTrue(coeff >= -6 * self.sigma and coeff <= 6 * self.sigma)

    def test_secret_key_gen(self):
        secret_key = self.bfv.SecretKeyGen()
        # Ensure that the degree of the sample polynomial is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(secret_key.coefficients) <= self.bfv.n)

        # Ensure that the coefficients of the key are within {-1, 0, 1}
        for coeff in secret_key.coefficients:
            self.assertTrue(coeff == -1 or coeff == 0 or coeff == 1)

    def test_public_key_gen(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.SampleFromErrorDistribution()
        public_key = self.bfv.PublicKeyGen(secret_key, e)

        self.assertIsInstance(public_key, tuple)
        self.assertEqual(len(public_key), 2)
        self.assertIsInstance(public_key[0], Polynomial)
        self.assertIsInstance(public_key[1], Polynomial)

        # Ensure that the coefficients of the public key are within Z_q = (-q/2, q/2]
        for coeff in public_key[0].coefficients:
            self.assertTrue(coeff > -self.q // 2 and coeff <= self.q // 2)
        for coeff in public_key[1].coefficients:
            self.assertTrue(coeff > -self.q // 2 and coeff <= self.q // 2)

        # Ensure that the public key is a polynomial in Rq
        self.assertEqual(public_key[0].ring, self.bfv.Rq)
        self.assertEqual(public_key[1].ring, self.bfv.Rq)

    def test_message_sample(self):
        message = self.bfv.Rt.sample_polynomial()

        # Ensure that message is a polynomial in Rt
        self.assertEqual(message.ring, self.bfv.Rt)

    def test_wrong_message_sample(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.SampleFromErrorDistribution()
        public_key = self.bfv.PublicKeyGen(secret_key, e)
        u = self.bfv.SampleFromTernaryDistribution()

        message = self.bfv.Rq.sample_polynomial()

        # message must be in Rt, but I sampled it from Rq. So it must throw an error.
        with self.assertRaisesRegex(AssertionError, "The message must be in Rt."):
            e0 = self.bfv.SampleFromErrorDistribution()
            e1 = self.bfv.SampleFromErrorDistribution()

            self.bfv.Encrypt(public_key, message, (e0, e1), u, self.delta)

    def test_valid_encryption(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.SampleFromErrorDistribution()
        public_key = self.bfv.PublicKeyGen(secret_key, e)

        message = self.bfv.Rt.sample_polynomial()

        e0 = self.bfv.SampleFromErrorDistribution()
        e1 = self.bfv.SampleFromErrorDistribution()
        u = self.bfv.SampleFromTernaryDistribution()

        ciphertext = self.bfv.Encrypt(public_key, message, (e0, e1), u, self.delta)

        # Ensure that the ciphertext is a polynomial in Rq
        self.assertEqual(ciphertext[0].ring, self.bfv.Rq)
        self.assertEqual(ciphertext[1].ring, self.bfv.Rq)

    def test_valid_decryption(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.SampleFromErrorDistribution()
        public_key = self.bfv.PublicKeyGen(secret_key, e)

        message = self.bfv.Rt.sample_polynomial()

        e0 = self.bfv.SampleFromErrorDistribution()
        e1 = self.bfv.SampleFromErrorDistribution()
        u = self.bfv.SampleFromTernaryDistribution()

        ciphertext = self.bfv.Encrypt(public_key, message, (e0, e1), u, self.delta)

        dec = self.bfv.Decrypt(secret_key, ciphertext, (e0, e1), e, u)

        # ensure that message and dec are of the same degree
        self.assertEqual(len(message.coefficients), len(dec.coefficients))

        # ensure that message and dec are of the same coefficients
        for i in range(len(message.coefficients)):
            self.assertEqual(message.coefficients[i], dec.coefficients[i])

    def test_eval_add(self):
        secret_key = self.bfv.SecretKeyGen()

        e = self.bfv.SampleFromErrorDistribution()

        public_key = self.bfv.PublicKeyGen(secret_key, e)

        message1 = self.bfv.Rt.sample_polynomial()
        message2 = self.bfv.Rt.sample_polynomial()
        message_sum = poly_add(message1.coefficients, message2.coefficients)
        message_sum = Polynomial(message_sum, self.bfv.Rt)

        e0 = self.bfv.SampleFromErrorDistribution()
        e1 = self.bfv.SampleFromErrorDistribution()
        u1 = self.bfv.SampleFromTernaryDistribution()
        error1 = (e0, e1)
        ciphertext1 = self.bfv.Encrypt(public_key, message1, error1, u1, self.delta)

        e0 = self.bfv.SampleFromErrorDistribution()
        e1 = self.bfv.SampleFromErrorDistribution()
        u2 = self.bfv.SampleFromTernaryDistribution()
        error2 = (e0, e1)
        ciphertext2 = self.bfv.Encrypt(public_key, message2, error2, u2, self.delta)

        ciphertext_sum = self.bfv.EvalAdd(ciphertext1, ciphertext2)

        # ciphertext_sum must be a tuple of two polynomials in Rq
        self.assertEqual(ciphertext_sum[0].ring, self.bfv.Rq)
        self.assertEqual(ciphertext_sum[1].ring, self.bfv.Rq)

        # decrypt ciphertext_sum
        # Note that the after performing EvalAdd, the error is the sum of the errors of the two ciphertexts
        # As demonstrated in equation 11 of the paper https://inferati.azureedge.net/docs/inferati-fhe-bfv.pdf
        e0_sum = poly_add(error1[0].coefficients, error2[0].coefficients)
        e0_sum = Polynomial(e0_sum, self.bfv.Rq)
        e1_sum = poly_add(error1[1].coefficients, error2[1].coefficients)
        e1_sum = Polynomial(e1_sum, self.bfv.Rq)
        e_sum = (e0_sum, e1_sum)

        u_sum = poly_add(u1.coefficients, u2.coefficients)
        u_sum = Polynomial(u_sum, self.bfv.Rq)

        dec = self.bfv.Decrypt(secret_key, ciphertext_sum, e_sum, e, u_sum)

        # ensure that message_sum and dec are the same
        for i in range(len(message_sum.coefficients)):
            self.assertEqual(message_sum.coefficients[i], dec.coefficients[i])

    def test_eval_const_add(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.SampleFromErrorDistribution()

        public_key = self.bfv.PublicKeyGen(secret_key, e)

        message1 = self.bfv.Rt.sample_polynomial()
        message2 = self.bfv.Rt.sample_polynomial()
        message_sum = poly_add(message1.coefficients, message2.coefficients)

        e0 = self.bfv.SampleFromErrorDistribution()
        e1 = self.bfv.SampleFromErrorDistribution()
        error = (e0, e1)
        u1 = self.bfv.SampleFromTernaryDistribution()

        ciphertext1 = self.bfv.Encrypt(public_key, message1, error, u1, self.delta)

        u2 = self.bfv.SampleFromTernaryDistribution()

        const_ciphertext = self.bfv.EncryptConst(public_key, message2, u2, self.delta)

        ciphertext_sum = self.bfv.EvalAdd(ciphertext1, const_ciphertext)

        # ciphertext_sum must be a tuple of two polynomials in Rq
        self.assertEqual(ciphertext_sum[0].ring, self.bfv.Rq)
        self.assertEqual(ciphertext_sum[1].ring, self.bfv.Rq)

        u_sum = poly_add(u1.coefficients, u2.coefficients)
        u_sum = Polynomial(u_sum, self.bfv.Rq)

        # decrypt ciphertext_sum
        dec = self.bfv.Decrypt(secret_key, ciphertext_sum, error, e, u_sum)

        # reduce the coefficients of message_sum by the t using the centered remainder
        for i in range(len(message_sum)):
            message_sum[i] = get_centered_remainder(message_sum[i], self.t)

        # ensure that message_sum and dec are the same
        for i in range(len(message_sum)):
            self.assertEqual(message_sum[i], dec.coefficients[i])


class TestBFVVWithCRT(unittest.TestCase):
    # The bigmodulus q is intepreted as a product of small moduli qis.
    def setUp(self):
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
        self.q = q
        self.n = 1024
        self.sigma = 3.2
        self.discrete_gaussian = DiscreteGaussian(self.sigma)
        self.t = 65537
        self.delta = int(math.floor(q.q / self.t))

    # In the CRT setting, small integers such as noise and key coefficients are drawn from the Error distribution and Ternary distribution respectively and are stored as single precision integers.
    # In this test, the public key is generated in the Rq basis and then transformed to the CRT basis.
    # The encryption operation is performed in the CRT basis (Rqi).
    # Eventually, the ciphertext is recovered in the Rq basis and compared with the ciphertext generated in the Rq basis.
    def test_valid_encryption(self):
        bfv_rq = BFV(self.n, self.q.q, self.t, self.discrete_gaussian)
        secret_key = bfv_rq.SecretKeyGen()
        e = bfv_rq.SampleFromErrorDistribution()
        u = bfv_rq.SampleFromTernaryDistribution()
        e0 = bfv_rq.SampleFromErrorDistribution()
        e1 = bfv_rq.SampleFromErrorDistribution()
        message = bfv_rq.Rt.sample_polynomial()
        public_key = bfv_rq.PublicKeyGen(secret_key, e)

        # Perform encryption in CRT basis
        c0_rqis = []
        c1_rqis = []

        # Transform public key to CRT basis
        pk0_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            public_key[0], self.q
        )
        pk1_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            public_key[1], self.q
        )

        for i in range(len(self.q.qis)):
            bfv_rqi = BFV(self.n, self.q.qis[i], self.t, self.discrete_gaussian)
            public_key_rqi = (pk0_rqis[i], pk1_rqis[i])
            ciphertext = bfv_rqi.Encrypt(
                public_key_rqi, message, (e0, e1), u, self.delta
            )
            c0_rqis.append(ciphertext[0])
            c1_rqis.append(ciphertext[1])

        # Recover ciphertext
        c0 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(c0_rqis, self.q)
        c1 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(c1_rqis, self.q)

        # Perform encryption in Rq
        ciphertext = bfv_rq.Encrypt(public_key, message, (e0, e1), u, self.delta)

        # Assert that the two ciphertexts are the same
        self.assertEqual(c0.coefficients, ciphertext[0].coefficients)
        self.assertEqual(c1.coefficients, ciphertext[1].coefficients)

    # In the CRT setting, The uniform elements `a` used to generate the public key are chosen directly in the CRT basis by drawing uniform value ai in Rqi.
    # The operations for encryption are implemented directly in the CRT basis.
    # The ciphertext is recovered in the Rq basis and then decrypted in the Rqi basis. The decrypted message is compared with the original message.
    def test_valid_decryption(self):
        bfv_rq = BFV(self.n, self.q.q, self.t, self.discrete_gaussian)

        secret_key = bfv_rq.SecretKeyGen()
        e = bfv_rq.SampleFromErrorDistribution()
        u = bfv_rq.SampleFromTernaryDistribution()
        e0 = bfv_rq.SampleFromErrorDistribution()
        e1 = bfv_rq.SampleFromErrorDistribution()
        message = bfv_rq.Rt.sample_polynomial()

        # Perform encryption in CRT basis
        c0_rqis = []
        c1_rqis = []

        for i in range(len(self.q.qis)):
            bfv_rqi = BFV(self.n, self.q.qis[i], self.t, self.discrete_gaussian)
            public_key_rqi = bfv_rqi.PublicKeyGen(secret_key, e)
            ciphertext = bfv_rqi.Encrypt(
                public_key_rqi, message, (e0, e1), u, self.delta
            )
            c0_rqis.append(ciphertext[0])
            c1_rqis.append(ciphertext[1])

        # Recover ciphertext
        c0 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(c0_rqis, self.q)
        c1 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(c1_rqis, self.q)

        # Perform decryption in Rq basis
        dec = bfv_rq.Decrypt(secret_key, (c0, c1), (e0, e1), e, u)

        # Assert that the two decryptions are the same
        self.assertEqual(dec.coefficients, message.coefficients)


if __name__ == "__main__":
    unittest.main()
