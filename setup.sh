#!/bin/bash

repo_url="https://github.com/karinemiras/evoman_framework.git"
random_name="QyZ-912495"

# Clone the repository
git clone "$repo_url" "$random_name"

# Check if the clone was successful
if [ $? -eq 0 ]; then
    echo "evoman_framework cloned successfully."
else
    echo "Failed to clone the evoman_framework repository."
    exit 1
fi

folder_to_copy="evoman"
cp -r "$random_name/$folder_to_copy" ./evoman_strategies/

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "Folder '$folder_to_copy' copied successfully."
else
    echo "Failed to copy folder '$folder_to_copy'."
    exit 1
fi

# Delete the cloned repository
rm -rf "$random_name"

# Check if the deletion was successful
if [ $? -eq 0 ]; then
    echo "Repository $random_name deleted successfully."
else
    echo "Failed to delete repository $random_name."
    exit 1
fi

echo "setup completed successfully."
