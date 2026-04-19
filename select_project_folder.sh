#!/bin/bash
# User-friendly project folder selection script

echo "============================================"
echo "📁 AIGUIBook Project Folder Selection"
echo "============================================"
echo ""

# List available folders
FOLDERS=()
COUNTER=1
echo "📁 Available folders:"
for item in */; do
    if [ -d "$item" ]; then
        # Remove trailing slash
        folder_name="${item%/}"
        # Skip hidden folders
        if [[ ! "$folder_name" == .* ]]; then
            FOLDERS+=("$folder_name")
            echo "   $COUNTER. $folder_name"
            ((COUNTER++))
        fi
    fi
done

if [ ${#FOLDERS[@]} -eq 0 ]; then
    echo "❌ No folders found!"
    exit 1
fi

echo ""
read -p "Select folder (1-${#FOLDERS[@]}) or enter custom path: " choice

if [[ "$choice" =~ ^[0-9]+$ ]]; then
    # Numeric selection
    idx=$((choice - 1))
    if [ $idx -ge 0 ] && [ $idx -lt ${#FOLDERS[@]} ]; then
        SELECTED="${FOLDERS[$idx]}"
    else
        echo "❌ Invalid selection"
        exit 1
    fi
elif [ -n "$choice" ]; then
    # Custom path
    if [ -d "$choice" ]; then
        SELECTED="$choice"
    else
        echo "❌ Path does not exist"
        exit 1
    fi
else
    echo "❌ No selection made"
    exit 1
fi

# Verify it's an AIGUIBook project
if [ -f "$SELECTED/audiobook_ai/core/epub_parser.py" ]; then
    echo ""
    echo "✅ Selected folder: $SELECTED"
    echo "   (AIGUIBook project detected)"
    echo ""
    echo "export PROJECT_FOLDER='$SELECTED'"
    echo "$SELECTED" > /tmp/selected_project_folder.txt
else
    echo "⚠️  Warning: Not a standard AIGUIBook project"
    echo ""
    echo "export PROJECT_FOLDER='$SELECTED'"
    echo "$SELECTED" > /tmp/selected_project_folder.txt
fi
