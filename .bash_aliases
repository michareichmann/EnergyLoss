ELOSS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
alias eloss='ipython -i $ELOSS_DIR/main.py -- $@'

