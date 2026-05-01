import unittest
import numpy as np
import os
from oxtacore.v3.learned_index import RecursiveModelIndex, LinearLearnedIndex

class TestV3RMI(unittest.TestCase):
    def test_rmi_integration(self):
        # Create a non-linear dataset (steps)
        # Indexes 0-9 -> Offset 0
        # Indexes 10-19 -> Offset 100
        offsets = []
        for i in range(100):
            if i < 50:
                offsets.append(i * 10)
            else:
                offsets.append(500 + (i-50) * 100) # Sudden jump in slope

        # Train Linear (should perform poorly)
        lin_model = LinearLearnedIndex()
        lin_model.train(offsets)

        # Train RMI (should perform better)
        rmi_model = RecursiveModelIndex(num_leaves=5)
        rmi_model.train(offsets)

        # Compare Errors
        lin_pred = lin_model.predict(75)
        rmi_pred = rmi_model.predict(75)
        actual = offsets[75] # 500 + 25*100 = 3000

        lin_err = abs(actual - lin_pred)
        rmi_err = abs(actual - rmi_pred)

        # print(f"Linear Error: {lin_err}, RMI Error: {rmi_err}")

        # RMI should be significantly better or at least handle the piecewise nature
        self.assertLessEqual(rmi_model.max_global_error, lin_model.max_error)

    def test_save_load(self):
        model = RecursiveModelIndex(num_leaves=2)
        offsets = [1, 2, 3, 100, 101, 102]
        model.train(offsets)

        filename = "test_rmi.lin"
        model.save(filename)

        loaded = RecursiveModelIndex()
        loaded.load(filename)

        self.assertEqual(loaded.num_leaves, 2)
        self.assertEqual(loaded.num_records, 6)
        self.assertEqual(loaded.predict(4), model.predict(4))

        os.remove(filename)

if __name__ == '__main__':
    unittest.main()
