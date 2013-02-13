#!/bin/sh

rm -rf build
mkdir build
cd build
cmake .. >/dev/null

cat << EOF

Now enter ./build:
$ cd build

Building:
$ make

Testing:
$ ctest

EOF
