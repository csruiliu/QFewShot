#!/usr/bin/env bash
DATADIR=data/miniImagenet/data

mkdir -p $DATADIR

gdown -O $DATADIR/miniImageNet.tar.gz "https://drive.google.com/uc?id=1GUDPuoH3JfbGR078vsuF5UFHuJTGXGFb&export=download"

tar -xvf $DATADIR/miniImageNet.tar.gz -C $DATADIR
rm -f $DATADIR/miniImageNet.tar.gz

mv $DATADIR/mini-imagenet-cache-train.pkl $DATADIR/train.pkl
mv $DATADIR/mini-imagenet-cache-val.pkl $DATADIR/valid.pkl
mv $DATADIR/mini-imagenet-cache-test.pkl $DATADIR/test.pkl

