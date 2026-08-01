#pragma once

#include <tuple>
#include <mutex>

// Max allowable output size, in tiles. Used to allocate global lock buffer per
// device for sync across threadblocks
#define MAX_TILES_C (1024 * 1024)
#define MAX_BARRIERS 1024
#define BARRIER_LOCKS_OFFSET MAX_TILES_C

// MoE expert scheduler state, after the barrier counters: [0] next ticket, [1]
// retired groups, [2 + group] ticket published to group. Self-resetting,
// zero-initialized with the rest of the buffer
#define MOE_MAX_GROUPS 64
#define MOE_SCHED_OFFSET (MAX_TILES_C + 2 * MAX_BARRIERS)
#define MOE_SCHED_INTS (2 + MOE_MAX_GROUPS)

// Workspace size
#define WORKSPACE_SIZE (16 * 1024 * 1024)

#define MAX_DEVICES 16
#define CC_OLD 1
#define CC_AMPERE 2
#define CC_ADA 3
#define CC_HOPPER 4
#define CC_BLACKWELL 5

// Singleton to manage context for each device. Stores device attributes and a
// large-enough lock buffer per device
class DevCtx {
 private:
  int num_sms[MAX_DEVICES] = {};
  int cc[MAX_DEVICES] = {};
  void* locks[MAX_DEVICES] = {};
  void* ws[MAX_DEVICES] = {};
  std::mutex mtx;

 public:
  static DevCtx& instance();
  int get_num_sms(int device);
  int get_cc(int device);
  void* get_ws(int device);
  int* get_locks(int device);

 private:
  DevCtx() = default;
  DevCtx(const DevCtx&) = delete;
  DevCtx& operator=(const DevCtx&) = delete;
};

int g_get_cc(int device);
int g_get_num_sms(int device);

void prepare_ctx(int device);