import os
import sys
import urllib
import setuptools.build_meta
from setuptools.command.install import install
from packaging.version import parse, Version

# @pierce - TODO: Update for proper release
BASE_WHEEL_URL = "https://github.com/piercefreeman/flash-attention/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"

class CustomBuildBackend(setuptools.build_meta._BuildMetaBackend):

    def build_wheel(self, wheel_directory, config_settings=None, metadata_directory=None):
        this_file_directory = os.path.dirname(os.path.abspath(__file__))
        print(f'This file is located in: {this_file_directory}')

        sys.argv = [
            *sys.argv[:1],
            *self._global_args(config_settings),
            *self._arbitrary_args(config_settings),
        ]
        with setuptools.build_meta.no_install_setup_requires():
            self.run_setup()

        print("OS", os.environ["FLASH_ATTENTION_WHEEL_URL"])
        print("config_settings", config_settings)
        print("metadata_directory", metadata_directory)
        raise ValueError

        print("Guessing wheel URL: ", wheel_url)
        
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)
            os.system(f'pip install {wheel_filename}')
            os.remove(wheel_filename)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().build_wheel(wheel_directory, config_settings, metadata_directory)


_BACKEND = CustomBuildBackend()  # noqa


get_requires_for_build_wheel = _BACKEND.get_requires_for_build_wheel
get_requires_for_build_sdist = _BACKEND.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _BACKEND.prepare_metadata_for_build_wheel
build_wheel = _BACKEND.build_wheel
build_sdist = _BACKEND.build_sdist

