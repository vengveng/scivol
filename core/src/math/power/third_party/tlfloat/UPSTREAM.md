# TLFloat headers

Version: 1.17.1.
Source: <https://github.com/shibatch/tlfloat>.
Pinned commit: `3d986fbac6998b2c4f579459468a9a98c90a80d5`.
License: Boost Software License 1.0, retained in `LICENSE.txt` and the headers.

The five upstream headers in `tlfloat/` are unmodified. `FILES.json` records their
hashes. They supply fixed-width Quad special functions and exceptional-range
arithmetic for the native double-double adapter. No TLFloat library or runtime
Python dependency is required. Wheels include the license and this notice.

`tlfloatconfig.hpp` is scivol's configuration of the upstream template. It selects
128-bit integer arithmetic when the compiler exposes `__SIZEOF_INT128__`; other
compilers use TLFloat's limb implementation. Native half/quad floating types are
not required. The internal test macro `SCIVOL_WIDE_PORTABLE_LIMBS` exercises that
fallback on hosts with native 128-bit integers. Do not copy a CMake-generated
configuration from one build host into portable wheels.
