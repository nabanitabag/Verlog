Set up SSH keys so you never need to enter passwords:

bash
# On your local machine, generate SSH key (if you don't have one)

ssh-keygen -t ed25519 -C "your_email@example.com"

Press Enter for all prompts (use default location, no passphrase for automation)

# Copy your public key to CHTC
ssh-copy-id ${CHTC_USER}@ap2001.chtc.wisc.edu

# Test - should connect without password
ssh ${CHTC_USER}@ap2001.chtc.wisc.edu