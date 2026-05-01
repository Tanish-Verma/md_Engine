#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="co24btech11023@10.2.4.21"
REMOTE_SOURCE="/home/co24btech11023/MD_ENGINE/*"
LOCAL_DESTINATION="/home/tanish/md_engine/"

rsync -avz -e "ssh -X" "${SSH_TARGET}:${REMOTE_SOURCE}" "$LOCAL_DESTINATION"
