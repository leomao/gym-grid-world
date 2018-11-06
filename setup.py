from setuptools import setup
from pkg_resources import get_distribution, DistributionNotFound


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


requirements = [
    'gym >= 0.9.6'
    'numpy'
]


pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
requirements.append(pillow_req)


setup(
    name='gym_grid_world',
    version='0.4.0',
    install_requires=requirements
)
