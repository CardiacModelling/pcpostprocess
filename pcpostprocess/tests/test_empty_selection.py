#!/usr/bin/env python3
import os
import tempfile
import unittest

from pcpostprocess.scripts.run_herg_qc import run


class TestEmptySelection(unittest.TestCase):
    """Regression test for issue #110.

    When all selected wells fail QC and ``include_failed_traces=False``,
    ``run()`` should return gracefully instead of raising ``KeyError``.
    """

    def test_all_wells_fail_qc_no_crash(self):
        data = os.path.join('test_data', '13112023_MW2_FF')
        if not os.path.isdir(data):
            self.skipTest('test_data not available')

        with tempfile.TemporaryDirectory() as tmp:
            # A01 is known to fail staircase QC in this dataset
            result = run(
                data_path=data,
                output_path=tmp,
                save_id='test',
                staircase_protocols={
                    'staircaseramp (2)_2kHz': 'staircaseramp',
                },
                wells=['A01'],
                include_failed_traces=False,
            )
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
