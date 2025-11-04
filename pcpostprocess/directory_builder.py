import datetime
import os
import sys

from ._version import __commit_id__, __version__


def get_git_revision_hash():
    """
    Get the hash for the git commit currently being used.

    @return The most recent commit hash or a suitable message

    """

    return __commit_id__


def get_build_type():
    if "dev" in __version__:
        return "Develop"
    else:
        return "Release"


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
        description_fout.write(f"Version: {__version__}\n")
        description_fout.write(f"Build type: {get_build_type()}\n")
        description_fout.write(f"Commit: {git_hash}\n")
        command = " ".join(sys.argv)
        description_fout.write(f"Command: {command}\n")
    return dirname

