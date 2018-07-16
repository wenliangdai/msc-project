#!/usr/bin/env bash

get_all=false
while [[ "$1" =~ ^- && ! "$1" == "--" ]]; do case $1 in
  -n | --name )
    shift; name=$1
    ;;
  -a | --all)
    get_all=true
    ;;
esac; shift; done
if [[ "$1" == '--' ]]; then shift; fi

echo "Getting ${name}..."

root="/home/wenlidai/sunets-reproduce/"
path="${root}${name}"

if [ ! -d $name ]; then
  mkdir $name
fi

if $get_all; then
  scp wise:$path ./$name
else
  scp wise:"${path}/saved_loss.p ${path}/saved_accuracy.p" ./$name
fi

echo "Done!"

