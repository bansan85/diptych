from setuptools import find_packages, setup

with open("README.md", mode="r", encoding="utf-8") as f:
    readme = f.read()

with open("LICENSE", mode="r", encoding="utf-8") as f:
    license_text = f.read()

setup(
    name="diptych",
    version="0.0.1",
    description=(
        "Detect multiple pages in image scanned, split them and apply OCR."
    ),
    long_description=readme,
    author="Vincent LE GARREC",
    author_email="github@le-garrec.fr",
    url="https://github.com/bansan85/diptych",
    license=license_text,
    packages=find_packages(exclude=("tests", "docs")),
)
