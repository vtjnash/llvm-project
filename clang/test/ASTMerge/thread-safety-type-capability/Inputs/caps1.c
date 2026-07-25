struct __attribute__((capability("mutex"))) Mutex {};
struct Mutex mu1, mu2;

typedef __attribute__((requires_capability(mu1))) void (*cb1)(void);

cb1 same;
cb1 differ_mutex;
cb1 differ_presence;
