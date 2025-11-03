#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from pcpostprocess import directory_builder


@contextmanager
def temp_cwd():
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(old_cwd)


def make_info_dict(lines):
    return {l.split(": ", 1)[0]: l.split(": ", 1)[1]
            for l in lines
            if len(l.split(": ", 1)) == 2}


class TestDirectoryBuilder(unittest.TestCase):
    def test_git_exists(self):
        # If git doesn't exist, other tests will fail
        exe = shutil.which("git")
        self.assertIsNotNone(exe)

    def test_directory_with_git(self):
        test_dir = directory_builder.setup_output_directory("test_output",
                                                            self.__class__.__name__)

        # Check that git commit is there
        with open(os.path.join(test_dir, "pcpostprocess_info.txt"), "r") as fin:
            info_file_contents = [l.strip() for l in fin.readlines()]

        info_dict = make_info_dict(info_file_contents)

        # REGEX to check if the relevant line contains a git commit
        self.assertTrue(bool(re.fullmatch(r"[0-9a-fA-F]{40}",
                                          info_dict["Commit"].split("-", 1)[0])))
        return

    def test_directory_with_no_git_history(self):
        with temp_cwd():
            test_dir = directory_builder.setup_output_directory("test_output",
                                                                self.__class__.__name__)

            with open(os.path.join(test_dir, "pcpostprocess_info.txt"), "r") as fin:
                info_file_contents = [l.strip() for l in fin.readlines()]

            info_dict = make_info_dict(info_file_contents)

            self.assertEqual(info_dict["Commit"], "No git history")
        return

    def test_directory_with_empty_git_history(self):
        with temp_cwd():
            subprocess.run(["git", "--init"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            test_dir = directory_builder.setup_output_directory("test_output",
                                                                self.__class__.__name__)

            with open(os.path.join(test_dir, "pcpostprocess_info.txt"), "r") as fin:
                info_file_contents = [l.strip() for l in fin.readlines()]

            info_dict = make_info_dict(info_file_contents)

        self.assertEqual(info_dict["Commit"], "No git history")
        return

    def test_directory_with_no_git_binary(self):
        with temp_cwd():

            # Temporarily and safely modify path so git won't be found
            with patch.dict("os.environ", {"PATH": ""}):
                test_dir = directory_builder.setup_output_directory("test_output",
                                                                    self.__class__.__name__)

                with open(os.path.join(test_dir, "pcpostprocess_info.txt"), "r") as fin:
                    info_file_contents = [l.strip() for l in fin.readlines()]

                info_dict = make_info_dict(info_file_contents)

                self.assertEqual(info_dict["Commit"], "git not found")



