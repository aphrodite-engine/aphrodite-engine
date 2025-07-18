
import os
from typing import Dict
import inspect
from weakref import WeakValueDictionary

class WeakSingleton(type):
    """
    A metaclass that creates a WeakSingleton instance. This ensures that only one instance of the class exists.
    WeakSingleton doesn't hold a strong reference to the instance, allowing it to be garbage collected when no longer in use.

    Attributes:
        _instances (Dict[type, object]): A dictionary to store the single instance of each class.
        _instances_argspec (Dict[type, object]): A dictionary to store the argument specifications of each instance.

    Methods:
        __call__(cls, *args, **kwargs):
            Creates or returns the single instance of the class. If the instance already exists, it checks that the 
            arguments used to create the instance are the same as the ones provided. Raises an assertion error if 
            the arguments differ.
    """
    # NOTE(kzawora): The instances are stored in a weakref dictionary, 
    # which allows the instances to be garbage collected when they are 
    # no longer in use. This is important for tests, where model runner 
    # can get constructed and destroyed multiple times, and we don't 
    # want to reuse the bucketing context from the previous instance.
    _instances: WeakValueDictionary[type, object] = WeakValueDictionary()
    _instances_argspec: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        argspec = inspect.getcallargs(super().__call__, args, kwargs)
        if cls not in cls._instances:
            # NOTE(kzawora): *DO NOT* assign to self._instances[cls] here,
            # we need this variable to to force a strong reference.
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            cls._instances_argspec[cls] = argspec
        assert cls._instances_argspec[cls] == argspec, "Singleton instance already initialized with different arguments"
        return cls._instances[cls]

def get_bucketing_context():
    use_exponential_bucketing = os.environ.get(
        'APHRODITE_EXPONENTIAL_BUCKETING', 'false').lower() == 'true'
    if use_exponential_bucketing:
        from aphrodite.hpu_extension.bucketing.exponential import (
            HPUExponentialBucketingContext as HPUBucketingContext)
    else:
        from aphrodite.hpu_extension.bucketing.linear import HPUBucketingContext

    return HPUBucketingContext