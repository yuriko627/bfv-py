import numpy as np

class PolynomialRing:
	# polynomial ring R = Z[x]/f(x) where f(x)=x^n+1
	# n is a power of 2
	# If the modulus Q > 1 is specified then the ring is R_Q = Z_Q[x]/f(x). Namely, the coefficients of the polynomials are in the set Z_Q = (-Q/2, Q/2]
	def __init__(self, n, modulus=None):
		# ensure that n is a power of 2
		assert n > 0 and (n & (n-1)) == 0, "n must be a power of 2"

		fx = [1] + [0] * (n-1) + [1]

		self.denominator = fx
		self.Q = modulus
		self.n = n

	def sample_polynomial(self):
		"""
		Sample polynomial a_Q from R_Q.
		"""
		# ensure that modulus is set
		if self.Q is None:
			raise AssertionError("The modulus Q must be set to sample a polynomial from R_Q")
		
		# range for np.random.randint
		lower_bound = -self.Q // 2  # included
		upper_bound = self.Q // 2 + 1  # excluded

		# generate random coefficients
		a_Q_coeff = np.random.randint(lower_bound, upper_bound, size=self.n)

		return Polynomial(a_Q_coeff, self)

class Polynomial:
	def __init__(self, coefficients, ring: PolynomialRing):
		self.ring = ring

		# apply redution to the ring
		remainder = reduce_coefficients(coefficients, self.ring)
		self.coefficients = remainder

def reduce_coefficients(coefficients, ring):
	# reduce (divide) coefficients by the denominator polynomial
	_, remainder = np.polydiv(coefficients, ring.denominator)

	# if the ring is R_Q, apply reduction by taking coeff mod Q
	if ring.Q is not None:
		for i in range(len(remainder)):
			remainder[i] = get_centered_remainder(remainder[i], ring.Q)
		
		# ensure that the coefficients are in the set Z_Q wich is defined as (-Q/2, Q/2]
		for coeff in remainder:
			assert coeff > -ring.Q // 2 and coeff <= ring.Q // 2, "coefficients must be in the set (-Q/2, Q/2]"


	return remainder

def get_centered_remainder(x, modulus):
    # The concept of the centered remainder is that after performing the modulo operation,
    # The result is in the set (-modulus/2, ..., modulus/2], rather than [0, ..., modulus-1].
    # If r is in range [0, modulus/2] then the centered remainder is r.
    # If r is in range [modulus/2 + 1, modulus-1] then the centered remainder is r - modulus.
    # If modulus is 7, then the field is {-3, -2, -1, 0, 1, 2, 3}.
    # 10 % 7 = 3. The centered remainder is 3.
    # 11 % 7 = 4. The centered remainder is 4 - 7 = -3.
    r = x % modulus
    return r if r <= modulus / 2 else r - modulus




