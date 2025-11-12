#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest

from pcpostprocess import directory_builder


store_output = False


def read_info_dict(path):
    """ Reads a pcpostproces_info.txt and returns it as a dict. """
    with open(path, 'r') as f:
        items = [line.strip().split(': ') for line in f.readlines()]
        return dict(line for line in items if len(line) == 2)


class TestDirectoryBuilder(unittest.TestCase):
    """ Tests the DirectoryBuilder class. """

    def test_directory_builder(self):
        # Test that a pcpostprocess_info.txt is written, with a commit hash

        with tempfile.TemporaryDirectory() as d:
            if store_output:    # pragma: no cover
                d = 'test_output'
            path = directory_builder.setup_output_directory(
                d, 'directory_builder')

            # Check that git commit is written
            info_dict = read_info_dict(
                os.path.join(path, 'pcpostprocess_info.txt'))
            self.assertRegex(info_dict['Commit'], r'^g[0-9a-fA-F]{9}$')


if __name__ == '__main__':  # pragma: no cover
    if '-store' in sys.argv:
        store_output = True
        sys.argv.remove('-store')
    else:
        print('Add -store to store output')
    unittest.main()

