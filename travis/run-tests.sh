#!/bin/sh

retval=0

if [ "$CI" = "true" ] && [ "$TRAVIS" = "true" ]
then
  git clone --depth 1 ssh://git@github.com/bansan85/ocr-book-travis.git -b "$TRAVIS_JOB_NAME""$TRAVIS_PYTHON_VERSION" || { echo "Failure git clone" && exit 1; }
fi

pylint {**,.}/*.py || { retval=1 && echo "Failure pylint"; }
flake8 {**,.}/*.py || { retval=1 && echo "Failure flake8"; }
prospector {**,.}/*.py || { retval=1 && echo "Failure prospector"; }
pycodestyle {**,.}/*.py || { retval=1 && echo "Failure pycodestyle"; }
mypy {**,.}/*.py || { retval=1 && echo "Failure mypy"; }
pytest --no-cov-on-fail || { retval=1 && echo "Failure pytest"; }
sed -i "s/^.*created at.*$//" cov_html/*.html || { retval=1 && echo "Failure sed"; }

if [ $retval -eq 0 ] && [ "$CI" = "true" ] && [ "$TRAVIS" = "true" ]
then
  rm -Rf ocr-book-travis/*
  cp -R pytest-junit.xml cov_html coverage.xml ocr-book-travis/
  cd ocr-book-travis || exit 1
  git config --global user.name "Travis"
  git config --global user.email "travis-ci@le-garrec.fr"
  git add .
  if [ -n "$(git diff-index --name-only HEAD --)" ]
  then
    git commit -m "$TRAVIS_COMMIT_MESSAGE" -m "Update from $TRAVIS_COMMIT"
    git push || { echo "Failure git push" && exit 1; }
  fi
  cd .. || exit 1
fi

exit $retval
