import random


class PolynomialRing:
    def __init__(self, n, modulus):
        """
        Initialize a polynomial ring R_modulus = Z_modulus[x]/f(x) where f(x)=x^n+1.
        - n is a power of 2.
        """

        assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"

        fx = [1] + [0] * (n - 1) + [1]

        self.denominator = fx
        self.modulus = modulus
        self.n = n

    def sample_polynomial(self):
        """
        Sample polynomial a_modulus from R_modulus.
        """

        # range for random.randint
        lower_bound = -self.modulus // 2  # included
        upper_bound = self.modulus // 2 + 1  # excluded

        # generate n random coefficients in the range [-modulus/2, modulus/2]
        coeffs = [random.randint(lower_bound, upper_bound) for _ in range(self.n)]

        return Polynomial(coeffs, self)

    def __eq__(self, other):
        if isinstance(other, PolynomialRing):
            return (
                self.denominator == other.denominator and self.modulus == other.modulus
            )
        return False


class Polynomial:
    def __init__(self, coefficients, ring: PolynomialRing):
        self.ring = ring

        # apply redution of the coefficients to the ring
        remainder = reduce_coefficients(coefficients, self.ring)

        self.coefficients = remainder


def reduce_coefficients(coefficients, ring):
    # reduce (divide) coefficients by the denominator polynomial
    _, remainder = poly_div(coefficients, ring.denominator)

    # apply further reduction by taking coeff mod modulus
    for i in range(len(remainder)):
        remainder[i] = get_centered_remainder(remainder[i], ring.modulus)

    # pad with zeroes at the beginning of the remainder to make it of size n
    remainder = [0] * (ring.n - len(remainder)) + remainder

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


def poly_div(dividend: list[int], divisor: list[int]):
    dividend = [int(x) for x in dividend]

    # Initialize quotient and remainder
    quotient = [0] * (len(dividend) - len(divisor) + 1)
    remainder = list(dividend)

    # Main division loop
    for i in range(len(quotient)):
        coeff = (
            remainder[i] // divisor[0]
        )  # Calculate the leading coefficient of quotient
        # turn coeff into an integer
        coeff = coeff
        quotient[i] = coeff

        # Subtract the current divisor*coeff from the remainder
        for j in range(len(divisor)):
            rem = remainder[i + j]
            rem -= divisor[j] * coeff
            remainder[i + j] = rem

    # Remove leading zeroes in remainder, if any
    while remainder and remainder[0] == 0:
        remainder.pop(0)

    return quotient, remainder


def poly_mul(poly1: list[int], poly2: list[int]):
    # The degree of the product polynomial is the sum of the degrees of the input polynomials
    result_degree = len(poly1) + len(poly2) - 1
    # Initialize the product polynomial with zeros
    product = [0] * result_degree

    # Multiply each term of the first polynomial by each term of the second polynomial
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            product[i + j] += poly1[i] * poly2[j]

    return product


def poly_add(poly1: list[int], poly2: list[int]):
    # The degree of the sum polynomial is the max of the degrees of the input polynomials
    result_degree = max(len(poly1), len(poly2))
    # Initialize the sum polynomial with zeros
    sum = [0] * result_degree

    for i in range(len(poly1)):
        sum[i + result_degree - len(poly1)] += poly1[i]

    for i in range(len(poly2)):
        sum[i + result_degree - len(poly2)] += poly2[i]

    return sum
