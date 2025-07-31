#pragma once

#define DEBUG_PRINT 0

#if DEBUG_PRINT
#define DPRINTF(fmt, ...) printf("%s:%d " fmt, __FILE__, __LINE__, ## __VA_ARGS__);
#define DPRINTF0(fmt, ...) if (threadIdx.x == 0) printf("%s:%d " fmt, __FILE__, __LINE__, ## __VA_ARGS__);
#define PRODUCER_DPRINTF0(fmt, ...) if (threadIdx.x == 0   && block0()) printf("%s:%d " fmt, __FILE__, __LINE__, ## __VA_ARGS__);
#define CONSUMER_DPRINTF0(fmt, ...) if (threadIdx.x == 128 && block0()) printf("%s:%d " fmt, __FILE__, __LINE__, ## __VA_ARGS__);
#else
#define DPRINTF(fmt, ...)
#define DPRINTF0(fmt, ...)
#define PRODUCER_DPRINTF0(fmt, ...)
#define CONSUMER_DPRINTF0(fmt, ...)
#endif
