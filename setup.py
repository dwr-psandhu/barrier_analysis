from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='barrier_analysis',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Barrier Impact Analysis ",
    license="MIT",
    author="Nicky Sandhu",
    author_email='psandhu@water.ca.gov',
    url='https://github.com/dwr-psandhu/barrier_analysis',
    packages=['barrier_analysis'],
    entry_points={
        'console_scripts': [
            'barrier_analysis=barrier_analysis.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='barrier_analysis',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
