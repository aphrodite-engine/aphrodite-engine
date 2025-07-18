# https://github.com/vllm-project/vllm/blob/6d1479ca4b5a3904b6c5b4a1d741dda43efdc289/use_existing_torch.py

import glob

requires_files = glob.glob('requirements/*.txt')
requires_files += ["pyproject.toml"]
for file in requires_files:
    print(f">>> cleaning {file}")
    with open(file) as f:
        lines = f.readlines()
    if "torch" in "".join(lines).lower():
        print("removed:")
        with open(file, 'w') as f:
            for line in lines:
                if 'torch' not in line.lower():
                    f.write(line)
                else:
                    print(line.strip())
    print(f"<<< done cleaning {file}")
    print()
