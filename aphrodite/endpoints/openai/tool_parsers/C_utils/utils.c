#include <string.h>
#include <stdlib.h>
#include <ctype.h>

char* find_common_prefix(const char* s1, const char* s2) {
    int min_length = strlen(s1) < strlen(s2) ? strlen(s1) : strlen(s2);
    char* prefix = (char*)malloc(min_length + 1);
    int prefix_len = 0;
    
    for (int i = 0; i < min_length; i++) {
        if (s1[i] == s2[i]) {
            prefix[prefix_len++] = s1[i];
        } else {
            break;
        }
    }
    prefix[prefix_len] = '\0';
    return prefix;
}

char* find_common_suffix(const char* s1, const char* s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    int min_length = len1 < len2 ? len1 : len2;
    char* suffix = (char*)malloc(min_length + 1);
    int suffix_len = 0;
    
    for (int i = 1; i <= min_length; i++) {
        if (s1[len1 - i] == s2[len2 - i] && !isalnum(s1[len1 - i])) {
            suffix[min_length - suffix_len - 1] = s1[len1 - i];
            suffix_len++;
        } else {
            break;
        }
    }
    
    // Shift the suffix to the beginning of the allocated memory
    if (suffix_len > 0) {
        memmove(suffix, suffix + (min_length - suffix_len), suffix_len);
    }
    suffix[suffix_len] = '\0';
    return suffix;
}

char* extract_intermediate_diff(const char* curr, const char* old) {
    char* suffix = find_common_suffix(curr, old);
    int curr_len = strlen(curr);
    int suffix_len = strlen(suffix);
    int prefix_len;
    char* result;
    
    // Create temporary strings without the suffix
    char* curr_no_suffix = (char*)malloc(curr_len + 1);
    strcpy(curr_no_suffix, curr);
    curr_no_suffix[curr_len - suffix_len] = '\0';
    
    char* old_no_suffix = (char*)malloc(strlen(old) + 1);
    strcpy(old_no_suffix, old);
    old_no_suffix[strlen(old) - suffix_len] = '\0';
    
    char* prefix = find_common_prefix(curr_no_suffix, old_no_suffix);
    prefix_len = strlen(prefix);
    
    // Extract the difference
    result = (char*)malloc(curr_len - prefix_len - suffix_len + 1);
    strncpy(result, curr_no_suffix + prefix_len, curr_len - prefix_len - suffix_len);
    result[curr_len - prefix_len - suffix_len] = '\0';
    
    // Clean up
    free(suffix);
    free(prefix);
    free(curr_no_suffix);
    free(old_no_suffix);
    
    return result;
}

int* find_all_indices(const char* string, const char* substring, int* count) {
    int string_len = strlen(string);
    int substr_len = strlen(substring);
    int capacity = 10;
    int* indices = (int*)malloc(capacity * sizeof(int));
    *count = 0;
    
    for (int i = 0; i <= string_len - substr_len; i++) {
        if (strncmp(string + i, substring, substr_len) == 0) {
            if (*count >= capacity) {
                capacity *= 2;
                indices = (int*)realloc(indices, capacity * sizeof(int));
            }
            indices[(*count)++] = i;
        }
    }
    
    return indices;
}