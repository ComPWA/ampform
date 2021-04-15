# Freeze all requirements for this Python version in a constraints file
# https://pip.pypa.io/en/stable/user_guide/#constraints-files

PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ -z "$PYTHON_VERSION" ]]; then
    echo -e "\e[31;1mERROR: environment variable PYTHON_VERSION needs to be defined!\e[0m"
    exit 1
fi

OUTPUT_FILE=.constraints/py$PYTHON_VERSION.txt
pip-compile \
    --extra dev \
    --upgrade \
    --no-annotate \
    -o "${OUTPUT_FILE}" &&
    # https://github.com/jazzband/pip-tools/issues/431#issuecomment-277300235
    sed -i -e 's/typing_extensions/typing-extensions/g' "${OUTPUT_FILE}" &&
    exit 0

exit 1 # if failure
