import Cython.Compiler.Options
from Cython.Build import cythonize
from setuptools import setup

Cython.Compiler.Options.annotate = True

infiles = []

infiles += [
    "aphrodite/engine/aphrodite_engine.py",
    "aphrodite/transformers_utils/detokenizer.py",
    "aphrodite/engine/output_processor/single_step.py",
    "aphrodite/outputs.py",
    "aphrodite/engine/output_processor/stop_checker.py",
]

infiles += [
    "aphrodite/core/scheduler.py",
    "aphrodite/sequence.py",
    "aphrodite/core/block_manager.py",
]

infiles += [
    "aphrodite/modeling/layers/sampler.py",
    "aphrodite/sampling_params.py",
    "aphrodite/utils.py",
]

setup(ext_modules=cythonize(infiles,
                            annotate=False,
                            force=True,
                            compiler_directives={
                                'language_level': "3",
                                'infer_types': True
                            }))

# example usage: python3 build_cython.py build_ext --inplace
