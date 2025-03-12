from setuptools import setup

setup(name='aphrodite_add_dummy_model',
      version='0.1',
      packages=['aphrodite_add_dummy_model'],
      entry_points={
          'aphrodite.general_plugins':
          ["register_dummy_model = aphrodite_add_dummy_model:register"]
      })
