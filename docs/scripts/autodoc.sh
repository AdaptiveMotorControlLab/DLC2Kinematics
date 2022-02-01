#!/bin/bash

startup_check() {
    if [[ ! -d .git ]]; then
        echo "This script has to be executed from the git root"
        exit 1
    fi
}

write_module() {
    echo "${1}"
    echo "========================"
    echo ""
    echo ".. automodule:: ${1}"
    echo "    :members:"
}

write_toctree() {
    echo ""
    echo ""
    echo ""
    echo ""
    echo ".. toctree::"
    echo "    :maxdepth: 2"
    echo "    :caption: Contents:"
    echo ""
}

add_toctree_entry() {
    echo "    _gen/${1}"
}

root=$(dirname $0)
output=$(dirname $root)/_gen
mkdir -p ${output}
index=docs/index.rst

startup_check

pandoc README.md -o $index
write_toctree >> $index
for fname in $(find kinematik -type f -iname "*.py" | grep -v __init__)
do
    module=`echo $fname | sed -e 's/\.py$//' -e 's/\//\./g'`
    rstname=$output/${module}.rst
    write_module ${module} > $rstname
    add_toctree_entry ${module} >> $index
    echo "wrote ${rstname}"
done


