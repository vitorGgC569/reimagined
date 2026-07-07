from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # The package can embed nsos_ext*.pyd runtime binaries as package data.
        # Mark wheels as platform-specific instead of py3-none-any.
        return True


setup(distclass=BinaryDistribution)
