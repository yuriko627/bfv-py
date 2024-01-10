from .polynomial import PolynomialRing, Polynomial, poly_mul, poly_add
from .discrete_gauss import DiscreteGaussian
import numpy as np


class BFV:
    def __init__(self, n, q, t, distribution: DiscreteGaussian):
        """
        Initialize a BFV instance starting from the parameters

        Parameters:
        - n: degree of the f(x) which is the denominator of the polynomial ring, must be a power of 2.
        - q: modulus q of the ciphertext space
        - t: modulus t of the plaintext space
        - distribution: Error distribution (e.g. Gaussian).
        """
        # Ensure that the modulus of the plaintext space is smaller than the modulus of the polynomial ring
        if t > q:
            raise ValueError(
                "The modulus of the plaintext space must be smaller than the modulus of the polynomial ring."
            )

        # Ensure that n is a power of 2
        assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"

        # Ensure that p and q are greater than 1
        assert q > 1, "modulus q must be > 1"
        assert t > 1, "modulus t must be > 1"

        # Ensure that t is a prime number
        assert self.is_prime(t), "modulus t must be a prime number"

        # Ensure that q and t are coprime
        assert np.gcd(q, t) == 1, "modulus q and t must be coprime"

        self.n = n
        self.Rq = PolynomialRing(n, q)
        self.Rt = PolynomialRing(n, t)
        self.distribution = distribution

    def SampleFromTernaryDistribution(self):
        """
        Sample a polynomial from the χ Ternary distribution.
        Namely, the coefficients are sampled uniformely from the ternary set {-1, 0, 1}. (coefficients are either of them)

        Returns: Sampled polynomial.
        """

        coefficients = np.random.choice([-1, 0, 1], size=self.n)

        return Polynomial(coefficients, self.Rq)

    def SampleFromErrorDistribution(self):
        """
        Sample a polynomial from the χ Error distribution.

        Returns: Sampled polynomial.
        """
        # Sample a polynomial from the Error distribution
        coefficients = self.distribution.sample(self.n)
        return Polynomial(coefficients, self.Rq)

    def SecretKeyGen(self):
        """
        Randomly generate a secret key.

        Returns: Generated secret key polynomial.
        """

        return self.SampleFromTernaryDistribution()

    def PublicKeyGen(self, secret_key: Polynomial, e: Polynomial):
        """
        Generate a public key from a given secret key.

        Parameters:
        - secret_key: Secret key.
        - e: error polynomial sampled from the distribution χ Error.

        Returns: Generated public key.
        """
        # Sample a polynomial a from Rq
        a = self.Rq.sample_polynomial()
        # a*s. The result will be in Rq
        mul = poly_mul(a.coefficients, secret_key.coefficients)

        mul = Polynomial(mul, self.Rq)

        # a*s + e. The result will be in Rq
        b = poly_add(mul.coefficients, e.coefficients)

        b = Polynomial(b, self.Rq)
        pk0 = b

        # pk1 = -a. The result will be in Rq
        pk1 = poly_mul(a.coefficients, [-1])
        pk1 = Polynomial(pk1, self.Rq)

        # public_key = (b, -a)
        public_key = (pk0, pk1)

        return public_key

    def Encrypt(
        self,
        public_key: (Polynomial, Polynomial),
        m: Polynomial,
        error: (Polynomial, Polynomial),
        u: Polynomial,
        delta: int,
    ):
        """
        Encrypt a given message m with a given public_key .

        Parameters:
        - public_key: Public key.
        - m: message.
        - error: tuple of error values used in encryption. These must be polynomial sampled from the distribution χ Error.
        - u: ephermeral key polynomial sampled from the distribution χ Ternary.
        - delta: delta = q/t

        Returns:
        ciphertext: Generated ciphertext.
        """
        # Ensure that the message is in Rt
        if m.ring != self.Rt:
            raise AssertionError("The message must be in Rt.")

        # Polynomials e0, e1 are sampled the distribution χ Error
        e0 = error[0]
        e1 = error[1]

        # Compute the ciphertext.
        # delta * m
        delta_m = poly_mul([delta], m.coefficients)
        # pk0 * u
        pk0_u = poly_mul(public_key[0].coefficients, u.coefficients)

        # delta * m + pk0 * u + e0
        ct_0 = poly_add(delta_m, pk0_u)
        ct_0 = poly_add(ct_0, e0.coefficients)

        # ct_0 will be in Rq
        ct_0 = Polynomial(ct_0, self.Rq)

        # pk1 * u
        pk1_u = poly_mul(public_key[1].coefficients, u.coefficients)

        # pk1 * u + e1
        ct_1 = poly_add(pk1_u, e1.coefficients)

        # The result will be in Rq
        ct_1 = Polynomial(ct_1, self.Rq)

        ciphertext = (ct_0, ct_1)

        return ciphertext

    def EncryptConst(
        self,
        public_key: (Polynomial, Polynomial),
        m: Polynomial,
        u: Polynomial,
        delta: int,
    ):
        """
        Encrypt a given message m with a given public_key setting e0 and e1 to 0. This is used for the constant multiplication and addition.

        Parameters:
        - public_key: Public key.
        - m: message.
        - u: ephermeral key polynomial sampled from the distribution χ Ternary.
        - delta: delta = q/t

        Returns:
        ciphertext: Generated ciphertext.
        """
        # Ensure that the message is in Rt
        if m.ring != self.Rt:
            raise AssertionError("The message must be in Rt.")

        # Compute the ciphertext.
        # delta * m
        delta_m = poly_mul([delta], m.coefficients)
        # pk0 * u
        pk0_u = poly_mul(public_key[0].coefficients, u.coefficients)

        # ct_0 = delta * m + pk0 * u
        ct_0 = poly_add(delta_m, pk0_u)

        # ct_0 will be in Rq
        ct_0 = Polynomial(ct_0, self.Rq)

        # ct_1 = pk1 * u
        ct_1 = poly_mul(public_key[1].coefficients, u.coefficients)

        # ct_0 will be in Rq
        ct_1 = Polynomial(ct_1, self.Rq)

        ciphertext = (ct_0, ct_1)

        return ciphertext

    def Decrypt(
        self,
        secret_key: Polynomial,
        ciphertext: (Polynomial, Polynomial),
        error: (Polynomial, Polynomial),
        e: Polynomial,
        u: Polynomial,
    ):
        """
        Decrypt a given ciphertext with a given secret key.

        Parameters:
        - secret_key: Secret key.
        - ciphertext: Ciphertext.
        - error: tuple of error values used in encryption. This is used when calculating that the noise is small enough to decrypt the message.
        - e: error polynomial sampled from the distribution χ Error. Used for public key generation. This is used when calculating that the noise is small enough to decrypt the message.
        - u: ephermeral key polynomial sampled from the distribution χ Ternary. Used for encryption. This is used when calculating that the noise is small enough to decrypt the message.

        Returns: Decrypted message.
        """
        # dec = round(t/q * ((ct0 + ct1*s) mod s)
        ct0 = ciphertext[0].coefficients
        ct1 = ciphertext[1].coefficients
        s = secret_key.coefficients
        t = self.Rt.modulus
        q = self.Rq.modulus

        ct1_s = poly_mul(ct1, s)

        # ct0 + ct1*s
        numerator_1 = poly_add(ct0, ct1_s)

        # Ensure that all the errors v < q/(2t) - 1/2
        # v = u * e + e0 + s * e1
        u_e = poly_mul(u.coefficients, e.coefficients)
        s_e1 = poly_mul(s, error[1].coefficients)

        v = poly_add(u_e, error[0].coefficients)
        v = poly_add(v, s_e1)

        # fresh error v is in Rq
        v = Polynomial(v, self.Rq)

        rt_Q = q % t

        threshold = q / (2 * t) - rt_Q / 2

        for v in v.coefficients:
            assert abs(v) < (
                threshold
            ), f"Noise {abs(v)} exceeds the threshold value {threshold}, decryption won't work"

        # Numerator 1 is in Rq.
        numerator_1 = Polynomial(numerator_1, self.Rq)

        numerator = poly_mul([t], numerator_1.coefficients)

        # For each coefficient of the numerator, divide it by q and round it to the nearest integer
        quotient = [round(coeff / q) for coeff in numerator]

        # trim leading zeros
        quotient = np.trim_zeros(quotient, "f")

        # quotient is in Rt
        quotient = Polynomial(quotient, self.Rt)

        return quotient

    def EvalAdd(
        self,
        ciphertext1: (Polynomial, Polynomial),
        ciphertext2: (Polynomial, Polynomial),
    ):
        """
        Add two ciphertexts.

        Parameters:
        - ciphertext1: First ciphertext.
        - ciphertext2: Second ciphertext.

        Returns:
        ciphertext_sum: Sum of the two ciphertexts.
        """
        # ct1_0 + ct2_0
        ct0 = poly_add(ciphertext1[0].coefficients, ciphertext2[0].coefficients)
        ct0 = Polynomial(ct0, self.Rq)

        # ct1_1 + ct2_1
        ct1 = poly_add(ciphertext1[1].coefficients, ciphertext2[1].coefficients)
        ct1 = Polynomial(ct1, self.Rq)

        return (ct0, ct1)

    def is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, n):
            if n % i == 0:
                return False
        return True
