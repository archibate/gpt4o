export PYTHONPATH=rplugin/python3
# python -m gpt4o.context_simplifier
grep -r unittest\\.main rplugin/python3/gpt4o | cut -f1 -d: | cut -f3,4 -d/ | cut -f1 -d. | sed s/\\//./ | xargs -n1 python -m
