import datetime
import os
import subprocess
import sys
from importlib.metadata import version


def get_git_revision_hash():
    """
    Get the hash for the git commit currently being used.

    @return The most recent commit hash or the string "No git history"

    """

    ret_string = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    if ret_string.startswith("fatal: not a git repository"):
        ret_string = "No git history"

    else:
        # Check if working tree has uncommitted changes
        is_clean = subprocess.call(["git", "diff-index", "--quiet", "HEAD", "--"]) == 0

        if is_clean:
            ret_string += "-clean"
        else:
            ret_string += "-dirty"

    return ret_string


def setup_output_directory(dirname: str = None, subdir_name: str = None):
    """
    Create an output directory if one doesn't already exist. Place an info
    file in this directory which lists the date/time created, the version of
    pcpostprocess, the command-line arguments provided, and the most recent git
    commit. The two parameters allow for a user specified top-level directory and
    a script-defined name for a subdirectory.

    @param Optional directory name
    @param Optional subdirectory name

    @return The path to the created file directory (String)
    """

    if dirname is None:
        if subdir_name:
            dirname = os.path.join("output", f"{subdir_name}")
        else:
            dirname = os.path.join("output", "output")

    if subdir_name is not None:
        dirname = os.path.join(dirname, subdir_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, "pcpostprocess_info.txt"), "w") as description_fout:
        git_hash = get_git_revision_hash()
        datetimestr = str(datetime.datetime.now())
        description_fout.write("pcpostprocess output "
                               "https://github.com/CardiacModelling/pcpostprocess\n")
        description_fout.write(f"Date: {datetimestr}\n")
        description_fout.write(f"Version: {version('pcpostprocess')}\n")
        description_fout.write(f"Commit {git_hash}\n")

        command = " ".join(sys.argv)
        description_fout.write(f"Command: {command}\n")
    return dirname

