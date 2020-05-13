#!/bin/bash

# list all files in the repo:
#git rev-list --objects --all

# store files to be deleted in filestoremove.dat:
#git rev-list --objects --all | fgrep data/pulsefish | awk '{print $2}' > filestoremove.dat

# remove all references to the files from the active history of the repo:
for i in $(<filestoremove.dat); do
    echo $i
    git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch $i" --prune-empty --tag-name-filter cat -- --all
done

# force all references to the file to be expired and purged from the packfile:
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --aggressive --prune=now

# push:
git push origin --force --all
git push origin --force --tags

# then rebase all or remove and clone anew!

# for details see:
# https://help.github.com/en/github/authenticating-to-github/removing-sensitive-data-from-a-repository

# https://stackoverflow.com/questions/11050265/remove-large-pack-file-created-by-git

# https://stackoverflow.com/questions/460331/git-finding-a-filename-from-a-sha1
