#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="co24btech11023@10.2.4.21"
REMOTE_SOURCE="/home/co24btech11023/md_Engine/"
LOCAL_DESTINATION="md_Engine/"

rsync --exclude='.venv/' --exclude='venv/' --exclude='__pycache__/' --exclude='.git/' -avz -e "ssh -X" "${SSH_TARGET}:${REMOTE_SOURCE}" "$LOCAL_DESTINATION"
