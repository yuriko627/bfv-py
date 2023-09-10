from polynomial import PolynomialRing, Polynomial, custom_modulo
from discrete_gauss import DiscreteGaussian
import math
import numpy as np

class RLWE:
    def __init__(self, n, q, t, distribution: DiscreteGaussian):
        """
        Initialize the RLWE instance with a given polynomial ring and error distribution.

        Parameters:
        n: degree of the f(x) which is the denominator of the polynomial ring, must be a power of 2.
        q: modulus q
        t: modulus t of the plaintext space
        distribution: Error distribution (e.g. Gaussian).
        """
        # Ensure that the modulus of the plaintext space is smaller than the modulus of the polynomial ring
        if t > q:
            raise ValueError("The modulus of the plaintext space must be smaller than the modulus of the polynomial ring.")

        self.R = PolynomialRing(n)
        self.Rq = PolynomialRing(n, q)
        self.Rt = PolynomialRing(n, t)
        self.distribution = distribution

    def SampleFromChiKeyDistribution(self):
        """
        Sample a polynomial from the χ Key distribution.
        Namely, the coefficients are sampled uniformely from the ternary set {-1, 0, 1}. (coefficients are either of them)

    	Returns: Sampled polynomial.
        """

        # Sample n coefficients from the ternary set {-1, 0, 1}
        coefficients = np.random.choice([-1, 0, 1], size=self.R.n)

        return Polynomial(coefficients, self.R)

    def SampleFromChiErrorDistribution(self):
        """
        Sample a polynomial from the χ Error distribution.

        Returns: Sampled polynomial.
        """
        # Sample a polynomial from the χ Error distribution
        coefficients = self.distribution.sample(self.R.n)
        return Polynomial(coefficients, self.R)

    def SecretKeyGen(self):
        """
        Randomly generate a secret key.

        Returns: Generated secret key polynomial.
        """

        return self.SampleFromChiKeyDistribution()

    def PublicKeyGen(self, secret_key: Polynomial):
        """
        Generate a public key from a given secret key.

        Parameters:
        secret_key: Secret key.

        Returns: Generated public key.
        """
        # Sample a polynomial a from Rq
        a = self.Rq.sample_polynomial() # TODO: what is this distribution? is it correct?

        # Sample a polynomial e from the distribution χ Error
        e = self.SampleFromChiErrorDistribution() # TODO: what is this distribution? is it correct?

        # a*s. The result will be in Rq
        mul = np.polymul(a.coefficients, secret_key.coefficients)

        mul = Polynomial(mul, self.Rq)

        # a*s + e. The result will be in Rq
        b = np.polyadd(mul.coefficients, e.coefficients)

        b = Polynomial(b, self.Rq)

        pk0 = b

        # pk1 = -a. The result will be in Rq
        pk1 = np.polymul(a.coefficients, [-1])
        pk1 = Polynomial(pk1, self.Rq)

		# public_key = (b, -a)
        public_key = (pk0, pk1)

        return public_key

    def Encrypt(self, public_key: (Polynomial, Polynomial), m: Polynomial):
        """
        Encrypt a given message m with a given public_key .

        Parameters:
        public_key: Public key.
        m: message.

        Returns: Generated ciphertext.
        """
        # Ensure that the message is in Rt
        if m.ring != self.Rt:
            raise AssertionError("The message must be in Rt.")

        q = self.Rq.Q
        t = self.Rt.Q

		# Sample polynomials e0, e1 from the distribution χ Error
        e0 = self.SampleFromChiErrorDistribution()

        # Ensure that all the errors e < q/2t - 1/2
        for e in e0.coefficients:
            assert abs(e) < (q/2/t - 1/2), f"Error value of |e0|: {e} is too big, dycryption won't work"

        e1 = self.SampleFromChiErrorDistribution()

        # Ensure that all the errors e < q/2t - 1/2
        for e in e1.coefficients:
            assert abs(e) < (q/2/t - 1/2), f"Error value of |e1|: {e} is too big, dycryption won't work"

		# Sample polynomial u from the distribution χ Key
        u = self.SampleFromChiKeyDistribution()

		# delta = q/t
        delta = q / t

        # Round delta to the lower integer
        delta = math.floor(delta)

        # Compute the ciphertext.
		# delta * m
        delta_m = np.polymul(delta, m.coefficients)
		# pk0 * u
        pk0_u = np.polymul(public_key[0].coefficients, u.coefficients)

		# delta * m + pk0 * u + e0
        ct_0 = np.polyadd(delta_m, pk0_u)
        ct_0 = np.polyadd(ct_0, e0.coefficients)

        # ct_0 will be in Rq
        ct_0 = Polynomial(ct_0, self.Rq)

		# pk1 * u
        pk1_u = np.polymul(public_key[1].coefficients, u.coefficients)

		# pk1 * u + e1
        ct_1 = np.polyadd(pk1_u, e1.coefficients)

        # The result will be in Rq
        ct_1 = Polynomial(ct_1, self.Rq)

        ciphertext = (ct_0, ct_1)
        return ciphertext

    def Decrypt(self, secret_key: Polynomial, ciphertext: (Polynomial, Polynomial)):
        """
        Decrypt a given ciphertext with a given secret key.

        Parameters:
        secret_key: Secret key.
        ciphertext: Ciphertext.

        Returns: Decrypted message.
        """
        # dec = round(t/q * ((ct0 + ct1*s) mod s)
        ct0 = ciphertext[0].coefficients
        ct1_s = np.polymul(ciphertext[1].coefficients, secret_key.coefficients)

		# ct0 + ct1*s
        numerator_1 = np.polyadd(ct0, ct1_s)

        # Numerator 1 is in Rq.
        numerator_1 = Polynomial(numerator_1, self.Rq)

        t = self.Rt.Q
        q = self.Rq.Q

        numerator = np.polymul(t, numerator_1.coefficients)

        # For each coefficient of the numerator, divide it by q and round it to the nearest integer
        quotient = [round(coeff / q) for coeff in numerator]

        # resulting polynomial is in Rt
        quotient = Polynomial(quotient, self.Rt)

        return quotient