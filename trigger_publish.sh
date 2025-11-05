#!/bin/bash
set -e

# Check if gh is installed
if ! command -v gh &> /dev/null
then
    echo "gh could not be found, please install it to continue"
    exit 1
fi

# Get the repository name from the git remote
REPO=gueraf/flash-attention
# Get the current branch name
BRANCH=main

echo "Triggering publish workflow on $REPO for branch $BRANCH"

# Trigger the workflow
gh workflow run publish.yml --repo "$REPO" --ref "$BRANCH"
