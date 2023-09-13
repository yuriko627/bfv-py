import unittest
from discrete_gauss import DiscreteGaussian
from polynomial import PolynomialRing, Polynomial
from rlwe import RLWE

class TestRLWE(unittest.TestCase):

    def setUp(self):
        self.n = 4
        self.q = 700
        self.sigma = 10
        self.discrete_gaussian = DiscreteGaussian(self.sigma)
        self.t = 7
        # Note that t and q do not have to be prime nor coprime.
        self.rlwe = RLWE(self.n, self.q, self.t, self.discrete_gaussian)

    def test_rlwe_initialization(self):
        self.assertEqual(self.rlwe.R.denominator, PolynomialRing(self.n).denominator)
        self.assertEqual(self.rlwe.R.n, PolynomialRing(self.n).n)
        self.assertEqual(self.rlwe.R.Q, PolynomialRing(self.n).Q)
        self.assertEqual(self.rlwe.R.Z_Q, PolynomialRing(self.n).Z_Q)

        self.assertEqual(self.rlwe.Rq.denominator, PolynomialRing(self.n, self.q).denominator)
        self.assertEqual(self.rlwe.Rq.n, PolynomialRing(self.n, self.q).n)
        self.assertEqual(self.rlwe.Rq.Q, PolynomialRing(self.n, self.q).Q)
        self.assertEqual(self.rlwe.Rq.Z_Q, PolynomialRing(self.n, self.q).Z_Q)

        self.assertEqual(self.rlwe.Rt.denominator, PolynomialRing(self.n, self.t).denominator)
        self.assertEqual(self.rlwe.Rt.n, PolynomialRing(self.n, self.t).n)
        self.assertEqual(self.rlwe.Rt.Q, PolynomialRing(self.n, self.t).Q)
        self.assertEqual(self.rlwe.Rt.Z_Q, PolynomialRing(self.n, self.t).Z_Q)

        self.assertEqual(self.rlwe.distribution, self.discrete_gaussian)

    def test_sample_from_chi_key_distribution(self):
        key = self.rlwe.SampleFromChiKeyDistribution()
        # Ensure that the degree of the sampled poly is equal or less than d (it might be less if the leading coefficient sampled is 0)
        count = 0
        for coeff in key.coefficients:
            count += 1
        self.assertTrue(count <= self.rlwe.R.n)

        # Ensure that the key is a polynomial in ring R
        self.assertEqual(key.ring, self.rlwe.R)

        # Ensure that the coefficients of the key are within {-1, 0, 1}
        for coeff in key.coefficients:
            self.assertTrue(coeff == -1 or coeff == 0 or coeff == 1)

    def test_sample_from_chi_error_distribution(self):
        error = self.rlwe.SampleFromChiErrorDistribution()
        # Ensure that the degree of the sample polynomial is at most n-1, which means the has at most n coefficients
        count = 0
        for coeff in error.coefficients:
            count += 1
        self.assertTrue(count <= self.rlwe.R.n)

        # Ensure that the error is a polynomial in ring R
        self.assertEqual(error.ring, self.rlwe.R)

        # Ensure that the secret key is sampled from the error distribution by checking that each coefficient is within the range of the error distribution
        for coeff in error.coefficients:
            self.assertTrue(coeff >= -6*self.sigma and coeff <= 6*self.sigma)


    def test_secret_key_gen(self):
        secret_key = self.rlwe.SecretKeyGen()
        # Ensure that the degree of the sample polynomial is at most n-1, which means the has at most n coefficients
        count = 0
        for coeff in secret_key.coefficients:
            count += 1
        self.assertTrue(count <= self.rlwe.R.n)

        # Ensure that the secret key is a polynomial in ring R
        self.assertEqual(secret_key.ring, self.rlwe.R)

        # Ensure that the coefficients of the key are within {-1, 0, 1}
        for coeff in secret_key.coefficients:
            self.assertTrue(coeff == -1 or coeff == 0 or coeff == 1)

    def test_public_key_gen(self):
        secret_key = self.rlwe.SecretKeyGen()
        public_key = self.rlwe.PublicKeyGen(secret_key)

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
        self.assertEqual(public_key[0].ring, self.rlwe.Rq)
        self.assertEqual(public_key[1].ring, self.rlwe.Rq)

        # TODO: add public key's value check

    def test_message_sample(self):
        secret_key = self.rlwe.SecretKeyGen()
        public_key = self.rlwe.PublicKeyGen(secret_key)

        message = self.rlwe.Rt.sample_polynomial()

        # Ensure that message is a polynomial in Rt
        self.assertEqual(message.ring, self.rlwe.Rt)

    def test_wrong_message_sample(self):
        secret_key = self.rlwe.SecretKeyGen()
        public_key = self.rlwe.PublicKeyGen(secret_key)

        message = self.rlwe.Rq.sample_polynomial()

        # message must be in Rt, but I sampled it from Rq. So it must throw an error.
        with self.assertRaisesRegex(AssertionError, "The message must be in Rt."):
            self.rlwe.Encrypt(public_key, message)

    def test_valid_encryption(self):
        secret_key = self.rlwe.SecretKeyGen()
        public_key = self.rlwe.PublicKeyGen(secret_key)

        message = self.rlwe.Rt.sample_polynomial()

        ciphertext, error = self.rlwe.Encrypt(public_key, message)

        # Ensure that the coefficients of the public key are within Z_q = (-q/2, q/2]
        for coeff in ciphertext[0].coefficients:
            self.assertTrue(coeff > -self.q // 2 and coeff <= self.q // 2)
        for coeff in ciphertext[1].coefficients:
            self.assertTrue(coeff > -self.q // 2 and coeff <= self.q // 2)

        # Ensure that the ciphertext is a polynomial in Rq
        self.assertEqual(ciphertext[0].ring, self.rlwe.Rq)
        self.assertEqual(ciphertext[1].ring, self.rlwe.Rq)

        ## TODO: add ciphertext value check

    def test_valid_decryption(self):
        secret_key = self.rlwe.SecretKeyGen()
        public_key = self.rlwe.PublicKeyGen(secret_key)

        message = self.rlwe.Rt.sample_polynomial()

        ciphertext, error = self.rlwe.Encrypt(public_key, message)

        dec = self.rlwe.Decrypt(secret_key, ciphertext, error)

        # ensure that message and dec are of the same degree
        self.assertEqual(len(message.coefficients), len(dec.coefficients))

        # ensure that message and dec are of the same coefficients
        for i in range(len(message.coefficients)):
            self.assertEqual(message.coefficients[i], dec.coefficients[i])

if __name__ == "__main__":
    unittest.main()
