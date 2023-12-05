from polynomial import PolynomialRing, Polynomial
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

        # Ensure that n is a power of 2
        assert n > 0 and (n & (n-1)) == 0, "n must be a power of 2"

        # Ensure that p and q are greater than 1
        assert q > 1, "modulus q must be > 1"
        assert t > 1, "modulus t must be > 1"

        # Ensure that t is a prime number
        assert self.is_prime(t), "modulus t must be a prime number"

        self.R = PolynomialRing(n)
        self.Rq = PolynomialRing(n, q)
        self.Rt = PolynomialRing(n, t)
        self.distribution = distribution

        # Sample error polynomial from the distribution χ Error
        self.e = self.SampleFromChiErrorDistribution() ## used in public key gen

        # Sample ephemeral key polynomial u from the distribution χ Key
        self.u = self.SampleFromChiKeyDistribution()

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

        # a*s. The result will be in Rq
        mul = np.polymul(a.coefficients, secret_key.coefficients)

        # assert that the degree of mul is at most 2 * (n - 1). Namely, the number of coefficients is at most 2 * n
        assert len(mul) <= 2 * self.R.n, f"The degree of mul is {len(mul)} which is greater than 2 * (n - 1) = {2 * self.R.n}"

        mul = Polynomial(mul, self.Rq)

        # a*s + e. The result will be in Rq
        b = np.polyadd(mul.coefficients, self.e.coefficients)

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

        Returns:
        ciphertext: Generated ciphertext.
        error: tuple of error values used in encryption.
        """
        # Ensure that the message is in Rt
        if m.ring != self.Rt:
            raise AssertionError("The message must be in Rt.")

        q = self.Rq.Q
        t = self.Rt.Q

        # Sample polynomials e0, e1 from the distribution χ Error
        e0 = self.SampleFromChiErrorDistribution()

        e1 = self.SampleFromChiErrorDistribution()

        # delta = q/t
        delta = q / t

        # Round delta to the lower integer
        delta = math.floor(delta)

        # Compute the ciphertext.
        # delta * m
        delta_m = np.polymul(delta, m.coefficients)
        # pk0 * u
        pk0_u = np.polymul(public_key[0].coefficients, self.u.coefficients)

        # delta * m + pk0 * u + e0
        ct_0 = np.polyadd(delta_m, pk0_u)
        ct_0 = np.polyadd(ct_0, e0.coefficients)

        # ct_0 will be in Rq
        ct_0 = Polynomial(ct_0, self.Rq)

        # pk1 * u
        pk1_u = np.polymul(public_key[1].coefficients, self.u.coefficients)

        # pk1 * u + e1
        ct_1 = np.polyadd(pk1_u, e1.coefficients)

        # The result will be in Rq
        ct_1 = Polynomial(ct_1, self.Rq)

        ciphertext = (ct_0, ct_1)
        error = (e0, e1)

        return ciphertext, error
    
    def EncryptConst(self, public_key: (Polynomial, Polynomial), m: Polynomial):
        """
        Encrypt a given message m with a given public_key setting e0 and e1 to 0. This is used for the constant multiplication and addition.

        Parameters:
        public_key: Public key.
        m: message.

        Returns:
        ciphertext: Generated ciphertext.
        """
        # Ensure that the message is in Rt
        if m.ring != self.Rt:
            raise AssertionError("The message must be in Rt.")

        q = self.Rq.Q
        t = self.Rt.Q

        # delta = q/t
        delta = q / t

        # Round delta to the lower integer
        delta = math.floor(delta)

        # Compute the ciphertext.
        # delta * m
        delta_m = np.polymul(delta, m.coefficients)
        # pk0 * u
        pk0_u = np.polymul(public_key[0].coefficients, self.u.coefficients)

        # ct_0 = delta * m + pk0 * u 
        ct_0 = np.polyadd(delta_m, pk0_u)

        # ct_0 will be in Rq
        ct_0 = Polynomial(ct_0, self.Rq)

        # ct_1 = pk1 * u
        ct_1 = np.polymul(public_key[1].coefficients, self.u.coefficients)

        # ct_0 will be in Rq
        ct_1 = Polynomial(ct_1, self.Rq)

        ciphertext = (ct_0, ct_1)

        return ciphertext

    def Decrypt(self, secret_key: Polynomial, ciphertext: (Polynomial, Polynomial), error: (Polynomial, Polynomial)):
        """
        Decrypt a given ciphertext with a given secret key.

        Parameters:
        secret_key: Secret key.
        ciphertext: Ciphertext.
        error: tuple of error values used in encryption. This is used to ensure that the noise is small enough to decrypt the message.

        Returns: Decrypted message.
        """
        # dec = round(t/q * ((ct0 + ct1*s) mod s)
        ct0 = ciphertext[0].coefficients
        ct1 = ciphertext[1].coefficients
        s = secret_key.coefficients
        t = self.Rt.Q
        q = self.Rq.Q

        ct1_s = np.polymul(ct1, s)

        # ct0 + ct1*s
        numerator_1 = np.polyadd(ct0, ct1_s)

        # Ensure that all the errors v < q/(2t) - 1/2
        # v = u * e + e0 + s * e1
        u_e = np.polymul(self.u.coefficients, self.e.coefficients)
        s_e1 = np.polymul(s, error[1].coefficients)

        v = np.polyadd(u_e, error[0].coefficients)
        v = np.polyadd(v, s_e1)

        # fresh error v is in Rq
        v = Polynomial(v, self.Rq)

        threshold = q/(2*t) - 1/2

        for v in v.coefficients:
            assert abs(v) < (threshold), f"Noise {abs(v)} exceeds the threshold value {threshold}, decryption won't work"

        # Numerator 1 is in Rq.
        numerator_1 = Polynomial(numerator_1, self.Rq)

        numerator = np.polymul(t, numerator_1.coefficients)

        # For each coefficient of the numerator, divide it by q and round it to the nearest integer
        quotient = [round(coeff / q) for coeff in numerator]

        # trim leading zeros
        quotient = np.trim_zeros(quotient, 'f')

        # quotient is in Rt
        quotient = Polynomial(quotient, self.Rt)

        return quotient
    
    def EvalAdd(self, ciphertext1: (Polynomial, Polynomial), ciphertext2: (Polynomial, Polynomial)):
        """
        Add two ciphertexts.

        Parameters:
        ciphertext1: First ciphertext.
        ciphertext2: Second ciphertext.

        Returns:
        ciphertext_sum: Sum of the two ciphertexts.
        """
        # ct1_0 + ct2_0
        ct0 = np.polyadd(ciphertext1[0].coefficients, ciphertext2[0].coefficients)
        ct0 = Polynomial(ct0, self.Rq)

        # ct1_1 + ct2_1
        ct1 = np.polyadd(ciphertext1[1].coefficients, ciphertext2[1].coefficients)
        ct1 = Polynomial(ct1, self.Rq)

        return (ct0, ct1)

    def is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, n):
            if n % i == 0:
                return False
        return True
