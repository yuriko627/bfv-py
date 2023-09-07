import unittest
import numpy as np
from discrete_gauss import DiscreteGaussian

class TestDiscreteGaussian(unittest.TestCase):
	def SetUp(self):
		self.sigma = 3
		self.gaussian = DiscreteGaussian(self.sigma)

	def test_instantiation(self):
		# assigning correct value of sigma
		self.assertEqual(self.gaussian.sigma, self.sigma)

	def test_probability_distribution(self):
		# ensure probabilities sum up to approximately 1 (due to floating-poitn precision)
		self.assertAlmostEqual(self.gaussian.prob.sum())

	def test_sampling(self):
		# sample a large number of times and check if the mean is approximately 0 (exptected for this setup)
		samples = self.gaussian.sample(100000)
		self.assertAlmostEqual(np.mean(samples), 0, places=1)

if __name__ == '__main__':
    unittest.main()