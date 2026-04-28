#ifndef CPUINFER_NUMA_COMPAT_H
#define CPUINFER_NUMA_COMPAT_H

#include <sched.h>

#if defined(KT_KERNEL_USE_NUMA)
#include <numa.h>
#endif

inline int kt_numa_node_of_cpu(int cpu) {
#if defined(KT_KERNEL_USE_NUMA)
  return numa_node_of_cpu(cpu);
#else
  (void)cpu;
  return 0;
#endif
}

inline int kt_numa_num_configured_nodes() {
#if defined(KT_KERNEL_USE_NUMA)
  return numa_num_configured_nodes();
#else
  return 1;
#endif
}

inline void kt_set_to_numa(int this_numa) {
#if defined(KT_KERNEL_USE_NUMA)
  struct bitmask* mask = numa_bitmask_alloc(numa_num_configured_nodes());
  numa_bitmask_setbit(mask, this_numa);
  numa_bind(mask);
  numa_bitmask_free(mask);
#else
  (void)this_numa;
#endif
}

#if !defined(KT_KERNEL_USE_NUMA)
#define numa_node_of_cpu kt_numa_node_of_cpu
#define numa_num_configured_nodes kt_numa_num_configured_nodes
#endif

#endif  // CPUINFER_NUMA_COMPAT_H
