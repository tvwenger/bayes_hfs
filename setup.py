from setuptools import find_packages, setup
import versioneer

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="bayes_hfs",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Bayesian Hyperfine Spectroscopy Models",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.11",
    license="MIT",
    url="https://github.com/tvwenger/bayes_hfs",
)
