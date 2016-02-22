#include "utils.h"

// Complement & Compare technique, see
// http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
int isPowerOfTwo (unsigned int x)
{
    return ((x != 0) && ((x & (~x + 1)) == x));
}
