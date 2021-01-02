#!/bin/bash
# Need bash for shopt

shopt -s globstar dotglob
for i in **/*.ini; do
  echo $i
  ini_linter.py "$i" || exit 1
done

exit 0
