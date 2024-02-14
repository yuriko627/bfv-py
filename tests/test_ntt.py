import unittest
import random
import galois
from bfv.ntt import ntt_poly_mul
from bfv.polynomial import poly_mul_naive

class TestNTT(unittest.TestCase):

    def test_ntt(self):
        p = 1152921504606584833
        k = 4
        coeffs = [random.randint(0, p - 1) for _ in range(2**k)]

        # Go from coefficients to NTT evaluations and back to coefficients and check if they are the same
        ntt_evals = galois.ntt(coeffs, 2**k, p)
        ntt_coeffs = galois.intt(ntt_evals, 2**k, p)
        for i in range(2**k):
            assert ntt_coeffs[i] == coeffs[i]


    def test_poly_mul_ntt(self):
        p = 1152921504606584833
        k = 4
        coeffs_1 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
        coeffs_2 = [random.randint(0, p - 1) for _ in range(2**(k-1))]

        # multiply the polynomials naively
        product_naive = poly_mul_naive(coeffs_1, coeffs_2)

        # reduce the coefficients modulo p
        for i in range(len(product_naive)):
            product_naive[i] = product_naive[i] % p
       
        # perform the NTT. Note the size is 2**k since we require at least 2**(k-1) + 2**(k-1) - 1 = 2**k - 1 coefficients for the product
        ntt_evals_1 = galois.ntt(coeffs_1, 2**k, p)
        ntt_evals_2 = galois.ntt(coeffs_2, 2**k, p)

        # perform convolution (the product is already performed modulo p in the NTT domain)
        ntt_product = [ntt_evals_1[i] * ntt_evals_2[i] for i in range(2**k)]

        # perform the inverse NTT
        product = galois.intt(ntt_product, 2**k, p)

        # trim the product to the correct length
        product = product[:len(coeffs_1) + len(coeffs_2) - 1]

        for i in range(len(product)):
            assert product[i] == product_naive[i]

    def test_ntt_poly_mul_compact(self):
        p = 1152921504606584833
        k = 4
        coeffs_1 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
        coeffs_2 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
    
        product = ntt_poly_mul(coeffs_1, coeffs_2, 2**k, p)

        # multiply the polynomials naively
        product_naive = poly_mul_naive(coeffs_1, coeffs_2)

        # reduce the coefficients modulo p
        for i in range(len(product_naive)):
            product_naive[i] = product_naive[i] % p

        for i in range(len(product)):
            assert product[i] == product_naive[i]

