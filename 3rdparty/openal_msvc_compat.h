#pragma once

// Clang's function-effect attributes become part of a member-pointer type, which
// older MSVC STL invoke traits do not recognize. Keep OpenAL's implementation and
// other attributes unchanged; these optional analysis annotations are not needed
// for its runtime behavior. Preinclude before any OpenAL declarations.
#include "openal-soft/common/opthelpers.h"

#undef NONBLOCKING
#define NONBLOCKING
