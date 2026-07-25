struct __attribute__((capability("mutex"))) Mutex {};
struct Mutex mu1, mu2;

// Differently spelled synonym for caps1.c's 'cb1': the same requirement, so
// 'same' must merge without a complaint.
typedef __attribute__((exclusive_locks_required(mu1))) void (*cb1)(void);
typedef __attribute__((requires_capability(mu2))) void (*cb2)(void);
typedef void (*plain)(void);

cb1 same;
cb2 differ_mutex;
plain differ_presence;
