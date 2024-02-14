import unittest
from bfv.discrete_gauss import DiscreteGaussian
from bfv.polynomial import PolynomialRing, Polynomial
from bfv.bfv import RLWE, BFV, BFVCrt
from bfv.crt import CRTModuli
from mpmath import *

class TestRLWE(unittest.TestCase):
    def setUp(self):
        self.n = 1024
        self.q = 1152921504606584833
        self.sigma = 3.2
        self.discrete_gaussian = DiscreteGaussian(self.sigma)
        self.t = 65537
        self.rlwe = RLWE(self.n, self.q, self.t, self.discrete_gaussian)

    def test_rlwe_initialization(self):
        self.assertEqual(
            self.rlwe.Rq.denominator, PolynomialRing(self.n, self.q).denominator
        )
        self.assertEqual(self.rlwe.Rq.n, PolynomialRing(self.n, self.q).n)
        self.assertEqual(self.rlwe.Rq.modulus, PolynomialRing(self.n, self.q).modulus)

        self.assertEqual(
            self.rlwe.Rt.denominator, PolynomialRing(self.n, self.t).denominator
        )
        self.assertEqual(self.rlwe.Rt.n, PolynomialRing(self.n, self.t).n)
        self.assertEqual(self.rlwe.Rt.modulus, PolynomialRing(self.n, self.t).modulus)

        self.assertEqual(self.rlwe.distribution, self.discrete_gaussian)

    def test_sample_from_chi_key_distribution(self):
        key = self.rlwe.SampleFromTernaryDistribution()
        # Ensure that the degree of the sampled poly is equal or less than d (it might be less if the leading coefficient sampled is 0)
        self.assertTrue(len(key.coefficients) <= self.rlwe.n)

        # Ensure that the coefficients of the key are within {-1, 0, 1}
        for coeff in key.coefficients:
            self.assertTrue(coeff == -1 or coeff == 0 or coeff == 1)

    def test_sample_from_chi_error_distribution(self):
        error = self.rlwe.SampleFromErrorDistribution()
        # Ensure that the degree of the sample polynomial is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(error.coefficients) <= self.rlwe.n)

        # Ensure that the secret key is sampled from the error distribution by checking that each coefficient is within the range of the error distribution
        for coeff in error.coefficients:
            self.assertTrue(coeff >= -6 * self.sigma and coeff <= 6 * self.sigma)


class TestBFV(unittest.TestCase):
    def setUp(self):
        n = 1024
        q = 1152921504606584833
        sigma = 3.2
        discrete_gaussian = DiscreteGaussian(sigma)
        t = 65537
        rlwe = RLWE(n, q, t, discrete_gaussian)
        self.bfv = BFV(rlwe)

    def test_secret_key_gen(self):
        secret_key = self.bfv.SecretKeyGen()
        # Ensure that the degree of the sample polynomial is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(secret_key.coefficients) <= self.bfv.rlwe.n)

        # Ensure that the coefficients of the key are within {-1, 0, 1}
        for coeff in secret_key.coefficients:
            self.assertTrue(coeff == -1 or coeff == 0 or coeff == 1)

    def test_public_key_gen(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()
        a = self.bfv.rlwe.Rq.sample_polynomial()
        public_key = self.bfv.PublicKeyGen(secret_key, e, a)

        self.assertIsInstance(public_key, tuple)
        self.assertEqual(len(public_key), 2)
        self.assertIsInstance(public_key[0], Polynomial)
        self.assertIsInstance(public_key[1], Polynomial)

        # Ensure that the coefficients of the public key are within the range [-(q-1)/2, (q-1)/2]
        lower_bound = -(self.bfv.rlwe.Rq.modulus - 1) / 2 # inclusive
        upper_bound = (self.bfv.rlwe.Rq.modulus - 1) / 2 # inclusive
        for coeff in public_key[0].coefficients:
            self.assertTrue(
                coeff >= lower_bound
                and coeff <= upper_bound
            )
        for coeff in public_key[1].coefficients:
            self.assertTrue(
                coeff >= lower_bound
                and coeff <= upper_bound
            )

    def test_message_sample(self):
        message = self.bfv.rlwe.Rt.sample_polynomial()

        # Ensure that the coefficients of the public key are within the range = [-(t-1)/2, (t-1)/2]
        lower_bound = -(self.bfv.rlwe.Rt.modulus - 1) / 2 # inclusive
        upper_bound = (self.bfv.rlwe.Rt.modulus - 1) / 2 # inclusive

        for coeff in message.coefficients:
            self.assertTrue(
                coeff >= lower_bound
                and coeff <= upper_bound
            )

    def test_valid_public_key_encryption(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()
        a = self.bfv.rlwe.Rq.sample_polynomial()
        public_key = self.bfv.PublicKeyGen(secret_key, e, a)

        message = self.bfv.rlwe.Rt.sample_polynomial()

        e0 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv.rlwe.SampleFromErrorDistribution()
        u = self.bfv.rlwe.SampleFromTernaryDistribution()

        ciphertext = self.bfv.PubKeyEncrypt(
            public_key, message, e0, e1, u
        )

        # Ensure that the ciphertext is a polynomial in Rq
        # Ensure that the coefficients of the ciphertext are within the range [-(q-1)/2, (q-1)/2]
        lower_bound = -(self.bfv.rlwe.Rq.modulus - 1) / 2 # inclusive
        upper_bound = (self.bfv.rlwe.Rq.modulus - 1) / 2 # inclusive

        for coeff in ciphertext[0].coefficients:
            self.assertTrue(
                coeff >= lower_bound
                and coeff <= upper_bound
            )
        for coeff in ciphertext[1].coefficients:
            self.assertTrue(
                coeff >= lower_bound
                and coeff <= upper_bound
            )
        # Ensure that the degree of the ciphertext is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(ciphertext[0].coefficients) <= self.bfv.rlwe.n)

    def test_valid_public_key_decryption(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()
        a = self.bfv.rlwe.Rq.sample_polynomial()
        public_key = self.bfv.PublicKeyGen(secret_key, e, a)

        message = self.bfv.rlwe.Rt.sample_polynomial()

        e0 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv.rlwe.SampleFromErrorDistribution()
        u = self.bfv.rlwe.SampleFromTernaryDistribution()

        ciphertext = self.bfv.PubKeyEncrypt(
            public_key, message, e0, e1, u
        )

        dec = self.bfv.Decrypt(secret_key, ciphertext)

        # ensure that message and dec are the same
        for i in range(len(message.coefficients)):
            self.assertEqual(message.coefficients[i], dec.coefficients[i])

    def test_valid_secret_key_decryption(self):
        secret_key = self.bfv.SecretKeyGen()
        message = self.bfv.rlwe.Rt.sample_polynomial()
        a = self.bfv.rlwe.Rq.sample_polynomial()
        e = self.bfv.rlwe.SampleFromErrorDistribution()

        ciphertext = self.bfv.SecretKeyEncrypt(secret_key, message, a, e)

        dec = self.bfv.Decrypt(secret_key, ciphertext)

        # ensure that message and dec are the same
        for i in range(len(message.coefficients)):
            self.assertEqual(message.coefficients[i], dec.coefficients[i])

    def test_eval_add(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()
        a = self.bfv.rlwe.Rq.sample_polynomial()
        public_key = self.bfv.PublicKeyGen(secret_key, e, a)

        message1 = self.bfv.rlwe.Rt.sample_polynomial()
        message2 = self.bfv.rlwe.Rt.sample_polynomial()
        message_sum = message1 + message2
        message_sum.reduce_in_ring(self.bfv.rlwe.Rt)

        e0_1 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1_1 = self.bfv.rlwe.SampleFromErrorDistribution()
        u_1 = self.bfv.rlwe.SampleFromTernaryDistribution()

        ciphertext1 = self.bfv.PubKeyEncrypt(
            public_key, message1, e0_1, e1_1, u_1
        )

        e0_2 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1_2 = self.bfv.rlwe.SampleFromErrorDistribution()
        u_2 = self.bfv.rlwe.SampleFromTernaryDistribution()

        ciphertext2 = self.bfv.PubKeyEncrypt(
            public_key, message2, e0_2, e1_2, u_2
        )

        ciphertext_sum = self.bfv.EvalAdd(ciphertext1, ciphertext2)

        # Ensure that the ciphertext_sum is a polynomial in Rq
        # Ensure that the ciphertext_sum of the ciphertext are within the range [-(q-1)/2, (q-1)/2]
        lower_bound = -(self.bfv.rlwe.Rq.modulus - 1) / 2 # inclusive
        upper_bound = (self.bfv.rlwe.Rq.modulus - 1) / 2 # inclusive

        for coeff in ciphertext_sum[0].coefficients:
            self.assertTrue(
                coeff >= lower_bound
                and coeff <= upper_bound
            )
        for coeff in ciphertext_sum[1].coefficients:
            self.assertTrue(
                coeff >= lower_bound
                and coeff <= upper_bound
            )
        # Ensure that the degree of the ciphertext_sum is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(ciphertext_sum[0].coefficients) <= self.bfv.rlwe.n)

        # decrypt ciphertext_sum
        dec = self.bfv.Decrypt(secret_key, ciphertext_sum)

        # ensure that message_sum and dec are the same
        for i in range(len(message_sum.coefficients)):
            self.assertEqual(message_sum.coefficients[i], dec.coefficients[i])

    def test_eval_const_add(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()
        a = self.bfv.rlwe.Rq.sample_polynomial()
        public_key = self.bfv.PublicKeyGen(secret_key, e, a)

        message1 = self.bfv.rlwe.Rt.sample_polynomial()
        message2 = self.bfv.rlwe.Rt.sample_polynomial()
        message_sum = message1 + message2
        message_sum.reduce_in_ring(self.bfv.rlwe.Rt)

        e0 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv.rlwe.SampleFromErrorDistribution()
        u_1 = self.bfv.rlwe.SampleFromTernaryDistribution()

        ciphertext1 = self.bfv.PubKeyEncrypt(
            public_key, message1, e0, e1, u_1)
        
        u_2 = self.bfv.rlwe.SampleFromTernaryDistribution()

        const_ciphertext = self.bfv.PubKeyEncryptConst(
            public_key, message2, u_2)

        ciphertext_sum = self.bfv.EvalAdd(ciphertext1, const_ciphertext)

        # decrypt ciphertext_sum
        dec = self.bfv.Decrypt(secret_key, ciphertext_sum)

        # ensure that message_sum and dec are the same
        for i in range(len(message_sum.coefficients)):
            self.assertEqual(message_sum.coefficients[i], dec.coefficients[i])

class TestBFVVWithCRT(unittest.TestCase):
    # The bigmodulus q is intepreted as a product of small moduli qis.
    # Note that here we are using a specific set of qis, in which each qi is a prime number.
    # This allows us to perform efficient polynomial multiplication leveraging NTT.
    def setUp(self):
        qis = [1152921504606584833,
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
            1152921504580894721]

        self.crt_moduli = CRTModuli(qis)
        self.n = 1024
        sigma = 3.2
        discrete_gaussian = DiscreteGaussian(sigma)
        t = 65537
        self.bfv_crt = BFVCrt(self.crt_moduli, self.n, t, discrete_gaussian)

    def test_valid_public_key_generation(self):

        s = self.bfv_crt.SecretKeyGen()
        e = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        ais = []
        for i in range(len(self.crt_moduli.qis)):
            ais.append(self.bfv_crt.bfv_qis[i].rlwe.Rq.sample_polynomial())

        pub_keys = self.bfv_crt.PublicKeyGen(s, e, ais)
        
        # Ensure that the coefficients of the public key are within the range [-(qi-1)/2, (qi-1)/2]
        for i, qi in enumerate(self.crt_moduli.qis):
            lower_bound = -(qi - 1) / 2
            upper_bound = (qi - 1) / 2
            for coeff in pub_keys[i][0].coefficients:
                self.assertTrue(coeff >= lower_bound and coeff <= upper_bound)
            for coeff in pub_keys[i][1].coefficients:
                self.assertTrue(coeff >= lower_bound and coeff <= upper_bound)

    def test_valid_public_key_decryption(self):
        s = self.bfv_crt.SecretKeyGen()
        e = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        ais = []
        for i in range(len(self.crt_moduli.qis)):
            ais.append(self.bfv_crt.bfv_qis[i].rlwe.Rq.sample_polynomial())

        pub_keys = self.bfv_crt.PublicKeyGen(s, e, ais)
        message = self.bfv_crt.bfv_q.rlwe.Rt.sample_polynomial()

        e0 = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        u = self.bfv_crt.bfv_q.rlwe.SampleFromTernaryDistribution()

        ciphertexts = self.bfv_crt.PubKeyEncrypt(pub_keys, message, e0, e1, u)

        message_prime = self.bfv_crt.Decrypt(s, ciphertexts)

        assert message_prime == message

    def test_valid_dummy_public_key_decryption(self):
        s = self.bfv_crt.SecretKeyGen()
        e = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        ais = []
        for i in range(len(self.crt_moduli.qis)):
            ais.append(self.bfv_crt.bfv_qis[i].rlwe.Rq.sample_polynomial())

        pub_keys = self.bfv_crt.PublicKeyGen(s, e, ais)
        message = self.bfv_crt.bfv_q.rlwe.Rt.sample_polynomial()

        e0 = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        u = self.bfv_crt.bfv_q.rlwe.SampleFromTernaryDistribution()

        ciphertexts = self.bfv_crt.PubKeyEncrypt(pub_keys, message, e0, e1, u)        
        message_prime = self.bfv_crt.DecryptDummy(s, ciphertexts)

        assert message_prime == message

    def test_valid_secret_key_decryption(self):
        s = self.bfv_crt.SecretKeyGen()
        e = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        ais = []
        for i in range(len(self.crt_moduli.qis)):
            ais.append(self.bfv_crt.bfv_qis[i].rlwe.Rq.sample_polynomial())

        message = self.bfv_crt.bfv_q.rlwe.Rt.sample_polynomial()

        ciphertexts = self.bfv_crt.SecretKeyEncrypt(s, ais, e, message)

        message_prime = self.bfv_crt.Decrypt(s, ciphertexts)

        assert message_prime == message

    def test_valid_dummy_secret_key_decryption(self):
        s = self.bfv_crt.SecretKeyGen()
        e = self.bfv_crt.bfv_q.rlwe.SampleFromErrorDistribution()
        ais = []
        for i in range(len(self.crt_moduli.qis)):
            ais.append(self.bfv_crt.bfv_qis[i].rlwe.Rq.sample_polynomial())

        message = self.bfv_crt.bfv_q.rlwe.Rt.sample_polynomial()

        ciphertexts = self.bfv_crt.SecretKeyEncrypt(s, ais, e, message)
        message_prime = self.bfv_crt.DecryptDummy(s, ciphertexts)

        assert message_prime == message