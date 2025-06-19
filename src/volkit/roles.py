from enum import Enum, auto, unique

@unique
class Role(Enum):
    MEAN        = auto()
    VOLATILITY  = auto()
    DENSITY     = auto()
    CORRELATION = auto()

CANONICAL_ORDER = [Role.MEAN, Role.VOLATILITY, Role.DENSITY,]

# print(CANONICAL_ORDER)
# print([role.name for role in CANONICAL_ORDER])
# print([role.value for role in CANONICAL_ORDER])
# print([role for role in CANONICAL_ORDER])
# print([type(role) for role in CANONICAL_ORDER])
