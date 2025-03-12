#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

char* find_common_prefix(const char* s1, const char* s2);
char* find_common_suffix(const char* s1, const char* s2);
char* extract_intermediate_diff(const char* curr, const char* old);
int* find_all_indices(const char* string, const char* substring, int* count);

#ifdef __cplusplus
}
#endif

#endif  // UTILS_H