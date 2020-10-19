#!/bin/sh

if [ "$CI" = "true" ] && [ "$TRAVIS" = "true" ]
then
  git clone --depth 1 ssh://git@github.com/bansan85/ocr-book-travis.git -b documentation || { echo "Failure git clone" && exit 1; }
  rm -Rf documentation/{*.htm*,*.png}
else
  mkdir -p ocr-book-travis
fi

cd doc || exit 1
find . -name "*.asciidoc" -exec asciidoctor {} -D ../ocr-book-travis \;
cp *.png ../ocr-book-travis/
cd .. || exit 1

if [ "$CI" = "true" ] && [ "$TRAVIS" = "true" ]
then
  cd ocr-book-travis || exit 1
  git config --global user.name "Travis"
  git config --global user.email "travis-ci@le-garrec.fr"
  git add .
  if [ -n "$(git diff-index --name-only HEAD --)" ]
  then
    git commit -m "$TRAVIS_COMMIT_MESSAGE" -m "Update from $TRAVIS_COMMIT"
    git push || exit 1
  fi
  cd .. || exit 1
fi

exit 0
