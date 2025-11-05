#!/bin/bash
set -e

# Check if gh is installed
if ! command -v gh &> /dev/null
then
    echo "gh could not be found, please install it to continue"
    exit 1
fi

# Get the repository name from the git remote
REPO=$(gh repo view --json name --jq .name)
# Get the current branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "Triggering publish workflow on $REPO for branch $BRANCH"

# Trigger the workflow
gh workflow run publish.yml --repo "$REPO" --ref "$BRANCH"
