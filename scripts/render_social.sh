#!/bin/sh
# Render scripts/social.html to site/assets/social.png: the repository's social preview on GitHub
# (Settings > General > Social preview) and the site's og:image. Needs Google Chrome.
set -e
cd "$(dirname "$0")/.."
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --hide-scrollbars --force-device-scale-factor=1 --window-size=1280,640 \
  --allow-file-access-from-files --virtual-time-budget=3000 \
  --screenshot="$PWD/site/assets/social.png" "file://$PWD/scripts/social.html"
