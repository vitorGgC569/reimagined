import unittest
import numpy as np
from oxtacore.v4_research.rmi_models import LinearModel, CubicSplineModel
from oxtacore.v4_research.hilbert import xy2d, d2xy
from oxtacore.v4_research.succinct import RankSelectBitVector
from oxtacore.v4_research.neuromorphic import LIFNeuron

class TestV4Research(unittest.TestCase):

    def test_rmi_linear(self):
        # Data: y = 2x + 1
        X = np.arange(10)
        Y = 2 * X + 1
        model = LinearModel()
        model.train(X, Y)
        pred = model.predict(5)
        self.assertAlmostEqual(pred, 11.0)

    def test_rmi_spline(self):
        # Data: y = x^3
        X = np.arange(10)
        Y = X ** 3
        model = CubicSplineModel()
        model.train(X, Y)
        pred = model.predict(3) # 27
        self.assertAlmostEqual(pred, 27.0, delta=0.1)

    def test_hilbert_curve(self):
        # 4x4 Grid (N=4)
        # (0,0) -> 0
        # (0,1) -> 1
        # (1,1) -> 2
        # (1,0) -> 3
        # Check mapping correctness
        n = 4
        d = xy2d(n, 0, 0)
        self.assertEqual(d, 0)
        x, y = d2xy(n, 0)
        self.assertEqual((x,y), (0,0))

        # Verify round-trip consistency
        for d_in in range(n*n):
            x, y = d2xy(n, d_in)
            d_out = xy2d(n, x, y)
            self.assertEqual(d_in, d_out, f"Round trip failed for d={d_in}")

    def test_succinct_rank_select(self):
        # 1 0 1 1 0
        # Rank1: 1 1 2 3 3
        bv = RankSelectBitVector("10110")
        self.assertEqual(bv.rank1(0), 1)
        self.assertEqual(bv.rank1(2), 2)
        self.assertEqual(bv.rank1(4), 3)

        # Select 2nd 1 -> index 2
        self.assertEqual(bv.select1(2), 2)

    def test_neuromorphic_lif(self):
        neuron = LIFNeuron(threshold=1.0, decay=0.5, rest=0.0)
        # Input 0.6 -> V=0.6
        s1 = neuron.step(0.6)
        self.assertEqual(s1, 0)
        # Input 0.6 -> V = 0.6*0.5 + 0.6 = 0.9
        s2 = neuron.step(0.6)
        self.assertEqual(s2, 0)
        # Input 0.6 -> V = 0.9*0.5 + 0.6 = 1.05 -> Spike!
        s3 = neuron.step(0.6)
        self.assertEqual(s3, 1)
        # Reset -> 0
        self.assertEqual(neuron.voltage, 0.0)

if __name__ == '__main__':
    unittest.main()
