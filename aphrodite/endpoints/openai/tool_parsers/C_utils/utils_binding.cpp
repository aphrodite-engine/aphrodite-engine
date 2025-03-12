#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Add this line for std::vector conversion
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(c_utils, m) {
  m.doc() = "C implementations of string utility functions";

  m.def(
      "find_common_prefix",
      [](const std::string& s1, const std::string& s2) {
        char* result = find_common_prefix(s1.c_str(), s2.c_str());
        std::string py_result(result);
        free(result);
        return py_result;
      },
      "Find common prefix between two strings");

  m.def(
      "find_common_suffix",
      [](const std::string& s1, const std::string& s2) {
        char* result = find_common_suffix(s1.c_str(), s2.c_str());
        std::string py_result(result);
        free(result);
        return py_result;
      },
      "Find common suffix between two strings");

  m.def(
      "extract_intermediate_diff",
      [](const std::string& curr, const std::string& old) {
        char* result = extract_intermediate_diff(curr.c_str(), old.c_str());
        std::string py_result(result);
        free(result);
        return py_result;
      },
      "Extract intermediate difference between two strings");

  m.def(
      "find_all_indices",
      [](const std::string& string, const std::string& substring) {
        int count;
        int* indices =
            find_all_indices(string.c_str(), substring.c_str(), &count);
        std::vector<int> result(indices, indices + count);
        free(indices);
        return result;
      },
      "Find all indices of substring in string");
}