#!/bin/bash
# Need bash for shopt

shopt -s globstar dotglob
for i in **/*.html; do
  echo "tidy $i"
  tidy -config ./.tidy-html5 "$i" | diff -pu "$i" - || exit 1
done

exit 0
