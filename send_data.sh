#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="co24btech11023@10.2.4.21"
LOCAL_SOURCE="/home/tanish/md_engine/"
REMOTE_DESTINATION="/home/co24btech11023/md_engine/"

rsync --exclude='.venv/' --exclude='venv/' -avz -e "ssh -X" "$LOCAL_SOURCE" "${SSH_TARGET}:${REMOTE_DESTINATION}"
