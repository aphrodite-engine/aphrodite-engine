class SamplerID:
    """A class that behaves like an enum but allows dynamic registration"""
    # Built-in samplers
    DRY = 7
    PENALTIES = 6
    NO_REPEAT_NGRAM = 8
    TEMPERATURE = 5
    TOP_NSIGMA = 9
    TOP_P_TOP_K = 0
    TOP_A = 1
    MIN_P = 2
    TFS = 3
    ETA_CUTOFF = 10
    EPSILON_CUTOFF = 11
    TYPICAL_P = 4
    QUADRATIC = 12
    XTC = 13

    _registry = {}

    def __init__(self, value: int):
        self.value = value

    @property
    def name(self) -> str:
        return self._registry.get(self.value, f"CUSTOM_{self.value}")

    def __eq__(self, other):
        if isinstance(other, SamplerID):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    @classmethod
    def register(cls, name: str, value: int) -> "SamplerID":
        if not name.isidentifier():
            raise ValueError(f"Invalid sampler name: {name}")
        if hasattr(cls, name):
            raise ValueError(f"Sampler ID {name} already exists")
        if value in cls._registry:
            raise ValueError(f"Sampler ID value {value} already in use")
            
        sampler_id = cls(value)
        cls._registry[value] = name
        setattr(cls, name, sampler_id)
        return sampler_id
