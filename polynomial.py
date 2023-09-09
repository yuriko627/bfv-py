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

		# if modulus is defined, chek that it is > 1
		if modulus is not None:
			assert modulus > 1, "modulus must be > 1"
			self.Z_Q = [j for j in range (-self.Q // 2 + 1, self. Q // 2 + 1)]
		else:
			self.Z_Q = None

	def sample_polynomial(self):
		"""
		Sample polynomial a_Q from U_Q: uniform distribution over R_Q.
		"""
		# ensure that modulus is set
		if self.Q is None:
			raise AssertionError("The modulus Q must be set to sample a polynomial from R_Q")

		a_Q_coeff = np.random.choice(self.Z_Q, size=self.n)

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
			remainder[i] = custom_modulo(remainder[i], ring.Q)

		# ensure that the coefficients are in the set Z_Q wich is defined as (-Q/2, Q/2]
		Z_Q_set = set(j for j in range(-ring.Q//2 + 1, ring.Q//2+1))
		for value in remainder:
			assert value in Z_Q_set, "Coefficients must be in Z_Q"

	return remainder

def custom_modulo(x, mod):
	# when mod = 7, the field is {-3, -2, -1, 0, 1, 2, 3}
	# if x = 10, mod = 7, then r = 3
	# if x = 11, mod = 7, then r = 4, which is divisible by 2 so final r will be 4 - 7 = -3
	r = x % mod
	return r if r <= mod / 2 else r - mod




