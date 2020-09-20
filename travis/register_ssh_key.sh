#!/bin/sh

if [ "$CI" = "true" ] && [ "$TRAVIS" = "true" ]
then
  # You must add options either openssl will says :
  # *** WARNING : deprecated key derivation used.
  openssl aes-256-cbc -md sha512 -pbkdf2 -iter 100000 -k "$PASSWORD_OPENSSL" -out travis/private-travis-ci-le-garrec.fr.key -in travis/private-travis-ci-le-garrec.fr.key.enc -d
  # ssh-add says:
  # Permissions 0664 for 'travis/github_travis-ci.key' are too open.
  # It is required that your private key files are NOT accessible by others.
  chmod 600 travis/private-travis-ci-le-garrec.fr.key
  ssh-add travis/private-travis-ci-le-garrec.fr.key
fi
