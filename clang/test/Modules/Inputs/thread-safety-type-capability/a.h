#include "mutexes.h"

// Identical annotations: these definitions must merge.
struct Same {
  typedef void (*cb)() __attribute__((requires_capability(mu1)));
  cb m;
};

// Same shape, different capability.
struct DiffMutex {
  typedef void (*cb)() __attribute__((requires_capability(mu1)));
  cb m;
};

// Same shape, one side annotated.
struct DiffPresence {
  typedef void (*cb)() __attribute__((requires_capability(mu1)));
  cb m;
};

// Same capability, different sharedness.
struct DiffSharedness {
  typedef void (*cb)() __attribute__((requires_capability(mu1)));
  cb m;
};

// Spelled with a synonym on the other side: these must still merge.
struct Synonym {
  typedef void (*cb)() __attribute__((requires_capability(mu1)));
  cb m;
};

// The try-acquire success value is spelled 'true' here and '1' on the other
// side: one value, so one type, so these must merge.
struct TryValue {
  typedef bool (*cb)() __attribute__((try_acquire_capability(true, mu1)));
  cb m;
};

// Genuinely different success values: these must not merge.
struct DiffTryValue {
  typedef bool (*cb)() __attribute__((try_acquire_capability(1, mu1)));
  cb m;
};
