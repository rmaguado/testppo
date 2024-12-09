#!/bin/bash

export DHOME="$HOME/projects/testppo"
export OUT="$PWD/runs"

if [ ! -e $DHOME/sjob.template ]; then
  echo "sjob.template not found"
  exit 1
fi

# NAME CONF GPUTYPE [W]

export NAME=$1
export CONF=$2
export GPUTYPE=${3:-L40S}

if [ "$GPUTYPE" != "A40" -a "$GPUTYPE" != "L40S" ]; then
  echo "$GPUTYPE not available"
  exit 1
fi

if [ ! -z $2 ]; then 
  printf "\nJob name: $NAME. Using $GPUTYPE \n"
  mkdir -p $OUT/$NAME
  envsubst '$NAME $CONF $GPUTYPE $OUT' < $PWD/sjob.template > $OUT/$NAME/$NAME.run
  sbatch $OUT/$NAME/$NAME.run

else
  printf "\nneed NAME CONF GPUTYPE(A40|L40S)\n\n"

fi