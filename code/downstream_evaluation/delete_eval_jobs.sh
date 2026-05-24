#!/bin/bash
set -euo pipefail

PROJECT="course-cs-552-jpazeved"

jobs=$(runai training list -p "$PROJECT" | grep eval-fuxi | awk '{print $1}')

if [[ -z "$jobs" ]]; then
    echo "No eval-fuxi jobs found."
    exit 0
fi

echo "Deleting jobs:"
echo "$jobs"
echo ""

while IFS= read -r job; do
    runai training delete "$job" -p "$PROJECT"
done <<< "$jobs"

echo "Done."
