#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="co24btech11023@10.2.4.21"
LOCAL_SOURCE="/home/tanish/md_Engine/"
REMOTE_DESTINATION="/home/co24btech11023/md_Engine/"

rsync --exclude='.venv/' --exclude='venv/' --exclude='__pycache__/' --exclude='.git/' -avz -e "ssh -X" "$LOCAL_SOURCE" "${SSH_TARGET}:${REMOTE_DESTINATION}"
