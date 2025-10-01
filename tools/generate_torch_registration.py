#!/usr/bin/env python3
"""
Script to generate PyTorch operation registration code from C++ function
signatures.

Usage:
    python generate_torch_registration.py input_file.h [--namespace my_ops] \
        [--backend torch::kCUDA]
"""

import re
import argparse
import sys
from typing import List, Tuple, Optional, Dict


class CPPToTorchMapper:
    """Maps C++ types to PyTorch schema types."""

    def __init__(self):
        # Mapping from C++ types to PyTorch schema types
        self.type_mapping = {
            # Tensor types
            'torch::Tensor&': 'Tensor!',
            'const torch::Tensor&': 'Tensor',
            'torch::Tensor': 'Tensor',
            'std::optional<torch::Tensor>&': 'Tensor!?',
            'const std::optional<torch::Tensor>&': 'Tensor?',
            'std::optional<torch::Tensor>': 'Tensor?',
            'std::vector<torch::Tensor>&': 'Tensor[]',  # Note: aliasing needs manual handling  # noqa: E501
            'const std::vector<torch::Tensor>&': 'Tensor[]',
            'std::vector<torch::Tensor>': 'Tensor[]',

            # Scalar types
            'int64_t': 'int',
            'const int64_t': 'int',
            'int64_t&': 'int',  # Unusual but possible
            'double': 'float',
            'const double': 'float',
            'float': 'float',
            'const float': 'float',
            'bool': 'bool',
            'const bool': 'bool',

            # String types
            'std::string': 'str',
            'const std::string&': 'str',
            'std::string&': 'str',

            # Vector types
            'std::vector<int64_t>': 'int[]',
            'const std::vector<int64_t>&': 'int[]',
            'std::vector<std::string>': 'str[]',
            'const std::vector<std::string>&': 'str[]',

            # PyTorch specific types
            'at::ScalarType': 'ScalarType',
            'const at::ScalarType': 'ScalarType',
            'std::optional<at::ScalarType>': 'ScalarType?',
            'const std::optional<at::ScalarType>&': 'ScalarType?',
        }

        # Common type aliases that might appear in headers
        self.type_aliases = {
            'c10::optional': 'std::optional',
        }

    def normalize_type(self, cpp_type: str) -> str:
        """Normalize C++ type string by removing extra spaces and applying
        aliases."""
        # Remove extra whitespace
        cpp_type = re.sub(r'\s+', ' ', cpp_type.strip())

        # Apply type aliases
        for alias, replacement in self.type_aliases.items():
            cpp_type = cpp_type.replace(alias, replacement)

        return cpp_type

    def cpp_to_schema_type(self, cpp_type: str) -> str:
        """Convert C++ type to PyTorch schema type."""
        normalized = self.normalize_type(cpp_type)

        if normalized in self.type_mapping:
            return self.type_mapping[normalized]

        # Handle some common variations
        if 'torch::Tensor' in normalized and 'optional' in normalized:
            if '&' in normalized and 'const' not in normalized:
                return 'Tensor!?'
            else:
                return 'Tensor?'
        elif 'torch::Tensor' in normalized and '&' in normalized and \
             'const' not in normalized:
            return 'Tensor!'
        elif 'torch::Tensor' in normalized and 'const' in normalized or \
             'torch::Tensor' in normalized:
            return 'Tensor'

        # Fallback - silently use a placeholder
        return 'Unknown'


class FunctionSignatureParser:
    """Parses C++ function signatures into structured data."""

    def __init__(self):
        self.mapper = CPPToTorchMapper()

    def parse_signature(self, signature: str) -> Optional[Dict]:
        """Parse a C++ function signature into components."""
        # Remove comments and normalize whitespace
        signature = re.sub(r'//.*$', '', signature, flags=re.MULTILINE)
        signature = re.sub(r'/\*.*?\*/', '', signature, flags=re.DOTALL)
        signature = re.sub(r'\s+', ' ', signature.strip())

        # Match function signature pattern
        # Return type, function name, parameters
        pattern = r'(\w+(?:\s*::\s*\w+)*(?:\s*<[^>]*>)?(?:\s*&)?)\s+(\w+)\s*\((.*?)\)\s*;?'  # noqa: E501
        match = re.match(pattern, signature, re.DOTALL)

        if not match:
            return None

        return_type, func_name, params_str = match.groups()

        # Parse parameters
        parameters = self.parse_parameters(params_str)

        return {
            'return_type': return_type.strip(),
            'function_name': func_name.strip(),
            'parameters': parameters,
            'raw_signature': signature
        }

    def parse_parameters(self, params_str: str) -> List[Tuple[str, str]]:
        """Parse parameter list into (type, name) tuples."""
        if not params_str.strip():
            return []

        parameters = []
        # Split by comma, but be careful of nested templates and function
        # pointers
        param_parts = self.split_parameters(params_str)

        for param in param_parts:
            param = param.strip()
            if not param:
                continue

            # Extract type and name
            # Handle cases like "const torch::Tensor& input", "int64_t size",
            # etc.
            # Look for the last identifier as the parameter name
            tokens = param.split()
            if len(tokens) >= 2:
                param_name = tokens[-1]
                param_type = ' '.join(tokens[:-1])
            else:
                # No parameter name provided, generate one
                param_type = param
                param_name = f"param_{len(parameters)}"

            parameters.append((param_type, param_name))

        return parameters

    def split_parameters(self, params_str: str) -> List[str]:
        """Split parameter string by commas, respecting nested templates."""
        parameters = []
        current_param = ""
        paren_depth = 0
        angle_depth = 0

        for char in params_str:
            if char == '<':
                angle_depth += 1
            elif char == '>':
                angle_depth -= 1
            elif char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0 and angle_depth == 0:
                parameters.append(current_param.strip())
                current_param = ""
                continue

            current_param += char

        if current_param.strip():
            parameters.append(current_param.strip())

        return parameters

class TorchRegistrationGenerator:
    """Generates PyTorch registration code from parsed function signatures."""

    def __init__(
        self, namespace: str = "my_ops",
        backend: str = "torch::kCUDA"
    ):
        self.namespace = namespace
        self.backend = backend
        self.mapper = CPPToTorchMapper()

    def generate_schema(self, func_info: Dict) -> str:
        """Generate PyTorch schema string from function info."""
        func_name = func_info['function_name']
        parameters = func_info['parameters']
        return_type = func_info['return_type']

        # Convert parameters to schema format
        schema_params = []
        for param_type, param_name in parameters:
            schema_type = self.mapper.cpp_to_schema_type(param_type)
            schema_params.append(f"{schema_type} {param_name}")

        # Determine return type
        if return_type == 'void':
            schema_return = '()'
        elif 'torch::Tensor' in return_type and 'vector' not in return_type.lower():
            schema_return = 'Tensor'
        elif 'std::vector<torch::Tensor>' in return_type:
            schema_return = 'Tensor[]'
        elif 'int64_t' in return_type:
            schema_return = 'int'
        elif 'bool' in return_type:
            schema_return = 'bool'
        elif 'string' in return_type.lower():
            schema_return = 'str'
        else:
            schema_return = f'Unknown /* {return_type} */'

        # Build schema string
        params_str = ', '.join(schema_params)
        schema = f"{func_name}({params_str}) -> {schema_return}"

        return schema

    def generate_registration(self, func_info: Dict) -> str:
        """Generate complete registration code for a function."""
        func_name = func_info['function_name']
        schema = self.generate_schema(func_info)

        # Split long schemas for readability
        if len(schema) > 80:
            def_code = self.format_multiline_def(func_name, schema)
        else:
            def_code = f'  ops.def("{schema}");'

        impl_code = f'  ops.impl("{func_name}", {self.backend}, &{func_name});'

        return f"{def_code}\n{impl_code}"

    def format_multiline_def(self, func_name: str, schema: str) -> str:
        """Format a long schema across multiple lines for ops.def()."""
        # Find the parameter list
        if '(' in schema and ')' in schema:
            func_part = schema[:schema.find('(') + 1]
            params_part = schema[schema.find('(') + 1:schema.rfind(')')]
            return_part = schema[schema.rfind(')'):]

            # Split parameters
            params = [p.strip() for p in params_part.split(',') if p.strip()]

            if len(params) <= 3:
                return f'  ops.def("{schema}");'

            # Format as proper C++ multiline string literal
            lines = []
            lines.append('  ops.def(')
            lines.append(f'      "{func_part}"')

            for i, param in enumerate(params):
                if i == len(params) - 1:
                    # Last parameter - add return part
                    lines.append(f'      "    {param}{return_part}");')
                else:
                    lines.append(f'      "    {param},"')

            return '\n'.join(lines)

        return f'  ops.def("{schema}");'

    def generate_library_block(self, func_infos: List[Dict]) -> str:
        """Generate complete TORCH_LIBRARY block."""
        header = f"TORCH_LIBRARY({self.namespace}, ops) {{\n"

        registrations = []
        for func_info in func_infos:
            registration = self.generate_registration(func_info)
            registrations.append(
                f"  // {func_info['function_name']}\n{registration}")

        body = '\n\n'.join(registrations)
        footer = "\n}"

        return header + body + footer

    def generate_raw_registrations(self, func_infos: List[Dict]) -> str:
        """Generate just the registration statements without wrapper."""
        registrations = []
        for func_info in func_infos:
            registration = self.generate_registration(func_info)
            registrations.append(registration)

        return '\n\n'.join(registrations)

def parse_header_file(filepath: str) -> List[str]:
    """Extract function signatures from a header file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)

    signatures = []
    lines = content.split('\n')
    current_signature = ""
    in_function = False

    for line in lines:
        line = line.strip()

        # Skip preprocessor directives, comments, and empty lines
        if (line.startswith('#') or line.startswith('//') or
            line.startswith('/*') or not line):
            continue

        # Skip common non-function declarations
        if (line.startswith('class ') or line.startswith('struct ') or
            line.startswith('namespace ') or line.startswith('using ') or
            line.startswith('typedef ') or line.startswith('template')):
            continue

        # Accumulate multi-line function signatures
        if ('(' in line and not line.endswith(';')) or in_function:
            in_function = True
            current_signature += " " + line
            if ';' in line:
                signatures.append(current_signature.strip())
                current_signature = ""
                in_function = False
        elif '(' in line and ';' in line:
            # Single-line function signature
            signatures.append(line)

    return signatures

def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch registration code from C++ function"
        " signatures."
    )
    parser.add_argument(
        "input_file", help="Input header file with C++ function signatures")
    parser.add_argument(
        "--namespace", default="my_ops", help="PyTorch library namespace")
    parser.add_argument(
        "--backend", default="torch::kCUDA", help="Backend for ops.impl")
    parser.add_argument(
        "--output", "-o", help="Output file (default: stdout)")
    parser.add_argument(
        "--with-library",
        action="store_true",
        help="Include TORCH_LIBRARY wrapper (default: raw statements only)")

    args = parser.parse_args()

    # Parse input file
    signatures = parse_header_file(args.input_file)

    if not signatures:
        sys.exit(1)

    # Parse signatures
    parser = FunctionSignatureParser()
    func_infos = []

    for sig in signatures:
        func_info = parser.parse_signature(sig)
        if func_info:
            func_infos.append(func_info)

    if not func_infos:
        sys.exit(1)

    # Generate registration code
    generator = TorchRegistrationGenerator(args.namespace, args.backend)
    if args.with_library:
        registration_code = generator.generate_library_block(func_infos)
    else:
        registration_code = generator.generate_raw_registrations(func_infos)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(registration_code)
    else:
        print(registration_code)


if __name__ == "__main__":
    main()
