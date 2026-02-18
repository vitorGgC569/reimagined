import unittest
import numpy as np
from oxtacore.v4_research.compression import pack_integers, unpack_integers, tANS
from oxtacore.v4_research.systems import IOUring, GDSDriver, CXLMemoryPool, PIMUnit

class TestV4Systems(unittest.TestCase):

    def test_bit_packing(self):
        # Pack 5-bit integers: 3, 31, 0, 15
        values = np.array([3, 31, 0, 15], dtype=np.int64)
        packed = pack_integers(values, 5)
        # 3 (00011) | 31 (11111)<<5 | 0 (00000)<<10 | 15 (01111)<<15
        # Verify first word
        unpacked = unpack_integers(packed, 4, 5)
        np.testing.assert_array_equal(values, unpacked)

    def test_tans_compression_ratio(self):
        # Symbol 'A' is 90%, 'B' is 10%.
        # Entropy: -0.9log(0.9) -0.1log(0.1) = 0.13 + 0.33 = 0.46 bits/symbol
        # 100 symbols -> ~46 bits.
        freqs = {'A': 90, 'B': 10}
        tans = tANS(freqs)
        data = ['A']*90 + ['B']*10
        bits = tans.encode(data)
        # Check it is better than 1 bit per symbol (100 bits)
        self.assertLess(bits, 100)

    def test_io_uring_flow(self):
        ring = IOUring()
        ring.submit_read(fd=1, buffer=None, offset=0, user_data=101)
        ring.submit_read(fd=1, buffer=None, offset=100, user_data=102)

        processed = ring.process_sq()
        self.assertEqual(processed, 2)

        cqe1 = ring.pop_cq()
        self.assertEqual(cqe1.user_data, 101)
        cqe2 = ring.pop_cq()
        self.assertEqual(cqe2.user_data, 102)

    def test_pim_operations(self):
        data = [1, 5, 2, 8, 3]
        pim = PIMUnit(data)
        # Scan > 4
        mask = pim.scan_filter(4)
        np.testing.assert_array_equal(mask, [False, True, False, True, False])
        # Sum
        self.assertEqual(pim.aggregate_sum(), 19)

if __name__ == '__main__':
    unittest.main()
