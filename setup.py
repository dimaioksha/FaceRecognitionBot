import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding="utf-8") as fh:
    reqr = fh.read()

setuptools.setup(
    name="Ioksha_X5",
    version="0.0.1",
    author="Dmitry Ioksha",
    author_email="enakentiy.lorens@yandex.ru",
    description="Final project X5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/DmitryIo/x5bot",
    packages=setuptools.find_packages(),
    install_requires=reqr,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)