from setuptools import setup, find_packages

setup(
		name='marlenvs',
		version='1.0.4',
		author='Lukas König',
		author_email='lukasmkoenig@gmx.net',
        packages=find_packages(),
		install_requires=['gym', 'numpy', 'wheel', 'pyglet'],
		decription="Multi-Agent Reinforcement Learning environments for gym."
	)
