#!/bin/bash
source .env

f=Verlog

cd ../.. # cd just outside the repo

# 1. Create the tarball
tar --exclude='.git' \
    --exclude='.idea'  \
    -czvf ${f}.tar.gz $f

USER=${CHTC_USER}
HOSTNAME="ap2001.chtc.wisc.edu"

echo "============================================"
echo "1. Transferring tarball to CHTC Staging..."
echo "============================================"
scp ${f}.tar.gz ${USER}@${HOSTNAME}:/staging/${USER}/

echo "============================================"
echo "2. Syncing chtc folder to CHTC Home directory..."
echo "============================================"
# This safely mirrors the folder (including .env) to your CHTC home directory
rsync -avz ${f} ${USER}@${HOSTNAME}:~/${f}/

# Clean up local tarball
rm ${f}.tar.gz

echo "Done! You can now run ./login_chtc.sh"