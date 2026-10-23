# Boost headers

Version: 1.89.0, release commit `ef7fea34711a189472893b88205b1dd3c275677b`.
Source: <https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.bz2>.
Archive SHA256:
`85a33fa22621b4f314f8e85e1a5e2a9363d22e4f4992925d4bb3bc631b5a0c7a`.

These unmodified headers provide double-double arithmetic and Bessel functions.
They are build-time inputs, not a runtime Python dependency. Wheels include the
license and this notice, not the header tree. `FILES.json` records file hashes.

The subset was extracted with the release's own `bcp` tool:

```sh
bcp --boost=/path/to/boost_1_89_0 --scan scivol/_native/wide_scalar.hpp scivol/_native/third_party/boost
```

`LICENSE_1_0.txt` is copied from the same archive. Keep upstream notices in each
header. Do not replace this portable subset with a single host's preprocessor
dependency trace.

BCP reports unresolved macro includes in obsolete C++03 type-trait branches;
scivol requires C++20. Its optional `BOOST_MATH_LOGGER_INCLUDE` target is absent
from the upstream release too, and instrumentation is not enabled. `<memory>` is
a standard-library header. These warnings do not justify changing upstream code.
Build and test the subset on each supported wheel platform when updating it.
