#!/bin/bash
# Need bash for shopt

shopt -s globstar dotglob
for i in {**/*.yml,**/*.yaml}; do
  echo "yamllint $i"
  yamllint "$i" || exit 1
done

exit 0
