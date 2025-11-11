#!/usr/bin/env python3
import os
import re
import tempfile
import unittest
from contextlib import ContextDecorator
from unittest.mock import patch

from pcpostprocess import directory_builder


class temp_cwd(ContextDecorator):
    def __enter__(self):
        self.old = os.getcwd()
        self.tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self.tmpdir.name)
        return self.tmpdir

    def __exit__(self, *exc):
        os.chdir(self.old)
        self.tmpdir.cleanup()
        return False


def make_info_dict(lines):
    return {l.split(": ", 1)[0]: l.split(": ", 1)[1]
            for l in lines
            if len(l.split(": ", 1)) == 2}


class TestDirectoryBuilder(unittest.TestCase):
    def test_directory_with_git(self):
        test_dir = directory_builder.setup_output_directory("test_output",
                                                            self.__class__.__name__)

        # Check that git commit is there
        with open(os.path.join(test_dir, "pcpostprocess_info.txt"), "r") as fin:
            info_file_contents = [l.strip() for l in fin.readlines()]

        info_dict = make_info_dict(info_file_contents)

        # REGEX to check if the relevant line contains a git commit
        self.assertTrue(bool(re.fullmatch(r"g[0-9a-fA-F]{9}",
                                          info_dict["Commit"])))
        return

    @temp_cwd()
    def test_directory_with_dev_version(self):
        with patch("pcpostprocess.directory_builder.__commit_id__", "a000"):
            test_dir = directory_builder.setup_output_directory("test_output",
                                                                self.__class__.__name__)

            with open(os.path.join(test_dir, "pcpostprocess_info.txt"), "r") as fin:
                info_file_contents = [l.strip() for l in fin.readlines()]

            info_dict = make_info_dict(info_file_contents)
            self.assertEqual(info_dict["Commit"], "a000")
        return

