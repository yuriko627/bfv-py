import numpy as np

class PolynnomialRing:
	# polynomial ring R = Z[x]/f(x) where f(x)=x^n+1
	# n is a power of 2
	# If the modulus Q > 1 is specified then the ring is R_Q = Z_Q[x]/f(x). Namely, the coefficients of the polynomials are in the set Z_Q = (-Q/2, Q/2]
	def __init__(self, n, modulus=None):
		# ensure that n is a power of 2
		assert n > 0 and (n & (n-1)) == 0, "n must be a power of 2"

		fx = [1] + [0] * (n-1) + [1]

		self.quotient = fx
		self.q = modulus
		self.n = n

		# if modulus is defined, chek that it is > 1
		if modulus is not None:
			assert modulus > 1, "modulus must be > 1"
			self.z_modulus = [j for j in range (-self.q // 2 + 1, self. q // 2 + 1)]
		else:
			self.z_modulus = None

	def sample_polynomial(self):
		"""
		Sample polynomial A_Q from R_Q, which is the distribution U_Q.
		"""
		# ensure that modulus is set
		if self.q is None:
			raise AssertionError("The modulus q must be set to sample a polynomial from R_Q")

		# sample a polynomial A_Q from R_Q
		A_Q_coeff = np. random.choice(self.z_modulus, size=self.n)

		return Polynomial(A_Q_coeff, self)




