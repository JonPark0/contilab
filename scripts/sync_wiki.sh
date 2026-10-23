#!/bin/bash
# Publish docs/wiki/*.md to the GitHub wiki (https://github.com/JonPark0/contilab/wiki).
# Relative links "Page.md" become "Page" (wiki link form); docs/wiki/README.md stays repo-only.
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
wiki_url=${WIKI_URL:-https://github.com/JonPark0/contilab.wiki.git}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone --depth 1 "$wiki_url" "$tmp/wiki" 2>/dev/null || git init -q -b master "$tmp/wiki"
find "$tmp/wiki" -maxdepth 1 -name '*.md' -delete
for f in "$repo_root"/docs/wiki/*.md; do
  name=$(basename "$f")
  [ "$name" = README.md ] && continue
  sed -E 's/\]\(([A-Za-z0-9_-]+)\.md\)/](\1)/g' "$f" > "$tmp/wiki/$name"
done
cd "$tmp/wiki"
git add -A
if git diff --cached --quiet; then echo "wiki already up to date"; exit 0; fi
git commit -q -m "Sync wiki from docs/wiki ($(git -C "$repo_root" rev-parse --short HEAD))"
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$wiki_url"
git push -q origin HEAD:master
echo "wiki published"
