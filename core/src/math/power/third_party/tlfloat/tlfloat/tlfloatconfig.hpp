// scivol configuration for TLFloat 1.17.1; see ../UPSTREAM.md.
#ifndef __TLFLOATVERSION_HPP_INCLUDED__
#define __TLFLOATVERSION_HPP_INCLUDED__

#define TLFLOAT_VERSION_MAJOR 1
#define TLFLOAT_VERSION_MINOR 17
#define TLFLOAT_VERSION_PATCH 1

// Use the compiler's integer extension when available, otherwise TLFloat's
// portable limb implementation. No native half/quad floating type is required.
#if defined(__SIZEOF_INT128__) && !defined(SCIVOL_WIDE_PORTABLE_LIMBS)
#define TLFLOAT_COMPILER_SUPPORTS_INT128
#endif

#endif
