from os import path
from setuptools import setup, find_namespace_packages


# Use READEME for long description.
root_dir = path.abspath( path.dirname( __file__ ) )
readme_path = path.join( root_dir, "README.md" )

with open( readme_path, encoding = "utf-8" ) as f:
    long_description = f.read()

# use requirements.txt for requirements.
req_path = path.join( root_dir, "requirements.txt" )

with open( req_path ) as f:
    requirements = f.read().split( '\n' )
    if '' in requirements:
        requirements.remove( '' )

setup( name = "qfast-qs",
       version = "2.0.0",
       description = "QFAST QSearch Plugin",
       long_description = long_description,
       long_description_content_type = "text/markdown",
       url = "https://github.com/BQSKit/qfast-qs",
       author = "Ed Younis",
       author_email = "edyounis@lbl.gov",
       classifiers = [
           "Development Status :: 5 - Production/Stable",
           "Environment :: Console",
           "Intended Audience :: Developers",
           "Intended Audience :: Science/Research",
           "Operating System :: OS Independent",
           "Programming Language :: Other Scripting Engines",
           "Programming Language :: Python :: 3 :: Only",
           "Programming Language :: Python :: 3.5",
           "Programming Language :: Python :: 3.6",
           "Programming Language :: Python :: 3.7",
           "Topic :: Scientific/Engineering",
           "Topic :: Scientific/Engineering :: Mathematics",
           "Topic :: Scientific/Engineering :: Physics",
           "Topic :: Software Development :: Compilers"
       ],
       keywords = "quantum synthesis compilation",
       project_urls = {
           "Bug Tracker": "https://github.com/BQSKit/qfast-qs/issues",
           "Source Code": "https://github.com/BQSKit/qfast-qs"
       },
       packages = [ "qfast.instantiation.native" ],
       install_requires = requirements,
       python_requires = ">=3.5, <4",
       zip_safe = False
)
