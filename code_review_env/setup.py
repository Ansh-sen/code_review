"""Setup script for the Code Review Environment package."""

from setuptools import setup, find_packages

setup(
    name="code-review-env",
    version="0.1.0",
    description="OpenEnv-compliant RL environment for AI code review",
    author="your-team-name",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.6.0",
    ],
)
