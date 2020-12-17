# https://github.com/jazzband/pip-tools/issues/625

PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

if [[ -z "$PYTHON_VERSION" ]]; then
    echo -e "\e[31;1mERROR: environment variable PYTHON_VERSION needs to be defined!\e[0m"
    exit 1
fi

mkdir -p reqs/$PYTHON_VERSION &&
    python reqs/extract_install_requires.py &&
    cp reqs/requirements*.in reqs/$PYTHON_VERSION/ &&
    rm reqs/$PYTHON_VERSION/requirements-dev.in &&
    pip-compile --upgrade \
        --no-annotate \
        reqs/requirements*.in \
        -o reqs/$PYTHON_VERSION/requirements-dev.txt &&
    for in_file in $(ls reqs/$PYTHON_VERSION/requirements*.in); do
        echo -e "-c requirements-dev.txt\n$(cat ${in_file})" >${in_file}
        out_file="${in_file/.in/.txt}"
        pip-compile "${in_file}" -o "${out_file}" --no-annotate
        # https://github.com/jazzband/pip-tools/issues/431#issuecomment-277300235
        sed -i -e 's/typing_extensions/typing-extensions/g' "${out_file}"
    done &&
    pip-sync reqs/$PYTHON_VERSION/requirements*.txt &&
    exit 0

exit 1 # if failure
