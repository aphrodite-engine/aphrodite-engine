#!/bin/bash
# This script tests if the python only compilation works correctly
# for users who do not have any compilers installed on their system

set -e
set -x

cd /aphrodite-workspace/

# uninstall aphrodite
pip3 uninstall -y aphrodite
# restore the original files
mv src/aphrodite ./aphrodite

# remove all compilers
apt remove --purge build-essential -y
apt autoremove -y

echo 'import os; os.system("touch /tmp/changed.file")' >> aphrodite/__init__.py

APHRODITE_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL=1 APHRODITE_USE_PRECOMPILED=1 pip3 install -vvv -e .

# Run the script
python3 -c 'import aphrodite'

# Check if the clangd log file was created
if [ ! -f /tmp/changed.file ]; then
    echo "changed.file was not created, python only compilation failed"
    exit 1
fi
