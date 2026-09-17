from setuptools import find_packages, setup

setup(
    name="hydrogen_res_opt",
    version="0.1.0",
    description=(
        "Co-optimization of RES capacity, substation selection and cable "
        "sizing for green hydrogen LCOH under stochastic DK1 electricity "
        "prices, with PowerFactory grid validation."
    ),
    author="Aalborg University - Department of Energy",
    packages=find_packages(exclude=["*.tests", "*.tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyomo>=6.7",
    ],
    extras_require={
        "dev": ["pytest>=7.4"],
    },
    entry_points={
        "console_scripts": [
            "hydrogen-res-opt=hydrogen_res_opt.cli:main",
        ],
    },
)
