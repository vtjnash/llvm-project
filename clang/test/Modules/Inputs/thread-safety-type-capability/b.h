#include "mutexes.h"

struct Same {
  typedef void (*cb)() __attribute__((requires_capability(mu1)));
  cb m;
};

struct DiffMutex {
  typedef void (*cb)() __attribute__((requires_capability(mu2)));
  cb m;
};

struct DiffPresence {
  typedef void (*cb)();
  cb m;
};

struct DiffSharedness {
  typedef void (*cb)() __attribute__((requires_shared_capability(mu1)));
  cb m;
};

struct Synonym {
  typedef void (*cb)() __attribute__((exclusive_locks_required(mu1)));
  cb m;
};
