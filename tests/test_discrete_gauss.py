import unittest
import numpy as np
from bfv.discrete_gauss import DiscreteGaussian


class TestDiscreteGaussian(unittest.TestCase):
    def setUp(self):
        self.sigma = 3.2
        self.gaussian = DiscreteGaussian(self.sigma)

    def test_instantiation(self):
        # assigning correct value of sigma
        self.assertEqual(self.gaussian.sigma, self.sigma)

    def test_probability_distribution(self):
        # ensure probabilities sum up to approximately 1 (due to floating-point precision)
        self.assertAlmostEqual(self.gaussian.prob.sum(), 1, places=5)

    def test_sampling(self):
        # sample a large number of times and check if the mean is approximately 0 (exptected for this setup)
        samples = self.gaussian.sample(100000)
        self.assertAlmostEqual(np.mean(samples), 0, places=1)
        # check that is doesn't sample values outside the range
        self.assertTrue(np.all(samples >= self.gaussian.z_lower))
        self.assertTrue(np.all(samples <= self.gaussian.z_upper))
        # check that all samples are integers
        self.assertTrue(np.all(np.equal(np.mod(samples, 1), 0)))


if __name__ == "__main__":
    unittest.main()
