#!/bin/bash

# Variables
URL="https://github.com/google/flatbuffers/releases/download/v25.2.10/Linux.flatc.binary.clang++-18.zip"
EXPECTED_HASH="6a20c2fc4e4e094574a0fd064f79a374eb9e6abba9e49d4543ec384b056725f6ca9f7823ba5952fcfa40e31a56a4e25baa659415d94edd69a7a978942577c579"
ZIP_FILE="flatc.linux.zip"

# Download the file
echo "Downloading $ZIP_FILE..."
wget -O "$ZIP_FILE" "$URL"

# Verify SHA-512 hash
echo "Verifying SHA-512 hash..."
CALCULATED_HASH=$(sha512sum "$ZIP_FILE" | awk '{print $1}')

if [ "$CALCULATED_HASH" == "$EXPECTED_HASH" ]; then
    echo "Hash verified successfully."
    echo "Unzipping $ZIP_FILE..."
    unzip "$ZIP_FILE"
else
    echo "Hash verification failed!"
    exit 1
fi

