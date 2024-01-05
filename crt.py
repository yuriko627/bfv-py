import math
from typing import List
from polynomial import Polynomial, PolynomialRing, get_centered_remainder


class Q:
    def __init__(self, qis: List[int]):
        """
        Initialize a Q object from the list of small moduli qis.
        """
        q = 1
        # qis should be same size single precision integers of 60 bits (or less)
        for qi in qis:
            assert isinstance(qi, int)
            assert qi.bit_length() <= 60, "qi is too large"
            q *= qi

        # qis should be coprime
        for i in range(len(qis)):
            for j in range(i + 1, len(qis)):
                assert math.gcd(qis[i], qis[j]) == 1, "qis are not pairwise coprime"

        self.qis = qis
        self.q = q


class CRTInteger:
    def __init__(self, q: Q, xis: List[int]):
        self.q = q
        self.xis = xis

    def from_crt_components(q: Q, xis: List[int]):
        """
        Initialize a CRTInteger object from the list of CRT components xis, which are integers in [0, qi) for each qi in qis, and the object Q, which contains the big modulus q and the list of small moduli qis.
        """
        assert len(xis) == len(q.qis), "xis and qis should have the same length"
        for i in range(len(xis)):
            assert xis[i] < q.qis[i], "xi should lie in [0, qi)"

        return CRTInteger(q, xis)

    def from_integer(q: Q, x: int):
        """
        Initialize a CRTInteger object from the integer x, which is in [0, q), and the object Q, which contains the big modulus q and the list of small moduli qis.
        """
        assert x < q.q, "x should lie in [0, q)"
        xis = []
        for qi in q.qis:
            xis.append(x % qi)

        return CRTInteger(q, xis)

    def recover(self):
        """
        Recover the integer x from its CRT representation. The integer x is in [0, q).
        """
        x = 0
        for i in range(len(self.q.qis)):
            xi = self.xis[i]
            qi_star = self.q.q // self.q.qis[i]
            qi_tilde = pow(
                qi_star, -1, self.q.qis[i]
            )  # inverse of qi_star mod self.q.qis[i]
            x += xi * qi_star * qi_tilde

        return x % self.q.q

    def recover_with_centered_remainder(self):
        """
        Recover the integer x from its CRT representation. The integer x is in (-q/2, q/2].
        """
        x = 0
        for i in range(len(self.q.qis)):
            xi = self.xis[i]
            qi_star = self.q.q // self.q.qis[i]
            qi_tilde = pow(
                qi_star, -1, self.q.qis[i]
            )  # inverse of qi_star mod self.q.qis[i]
            x += xi * qi_star * qi_tilde

        return get_centered_remainder(x, self.q.q)


class CRTPolynomial:
    def from_rq_polynomial_to_rqi_polynomials(rq_polynomial: Polynomial, q: Q):
        """
        Reduce polynomial `a` coefficients to its CRT representations

        Parameters:
        - rq_polynomial: polynomial in R_q
        - Q: object of type Q, which contains the big modulus q and the list of small moduli qis

        Returns: list of polynomials in R_qi
        """
        rqi_polynomials = []
        for qi in q.qis:
            rqi = PolynomialRing(rq_polynomial.ring.n, qi)
            rqis_polynomial = Polynomial(rq_polynomial.coefficients, rqi)
            rqi_polynomials.append(rqis_polynomial)

        return rqi_polynomials

    # Recover polynomial `a` from its CRT representations
    def from_rqi_polynomials_to_rq_polynomial(rqi_polynomials: List[Polynomial], q: Q):
        """
        Recover polynomial `a` from its CRT representations

        Parameters:
        - rqi_polynomials: list of polynomials in R_qi
        - Q: object of type Q, which contains the big modulus q and the list of small moduli qis

        Returns: polynomial in R_q
        """
        assert len(rqi_polynomials) == len(q.qis)
        rq_coefficients = []
        for i in range(len(rqi_polynomials[0].coefficients)):
            coeff_crt_components = []
            for j in range(len(rqi_polynomials)):
                coeff_crt_components.append(rqi_polynomials[j].coefficients[i])
            coeff_crt_integer = CRTInteger(q, coeff_crt_components)
            coeff = coeff_crt_integer.recover_with_centered_remainder()
            rq_coefficients.append(coeff)

        rq = PolynomialRing(rqi_polynomials[0].ring.n, q.q)
        return Polynomial(rq_coefficients, rq)
