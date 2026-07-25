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
