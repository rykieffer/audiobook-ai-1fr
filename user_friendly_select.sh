#!/bin/bash
# User-friendly folder selection for AIGUIBook project

echo "============================================"
echo "📁 AIGUIBook Project Folder Selection"
echo "============================================"
echo ""

# List folders with numbers
FOLDERS=()
COUNTER=1
echo "Available folders:"
echo ""
for item in */; do
    if [ -d "$item" ]; then
        folder_name="${item%/}"
        if [[ ! "$folder_name" == .* ]]; then
            FOLDERS+=("$folder_name")
            echo "  [$COUNTER] $folder_name"
            ((COUNTER++))
        fi
    fi
done

echo ""
read -p "Enter folder number or path: " choice

if [[ "$choice" =~ ^[0-9]+$ ]]; then
    idx=$((choice - 1))
    if [ $idx -ge 0 ] && [ $idx -lt ${#FOLDERS[@]} ]; then
        SELECTED="${FOLDERS[$idx]}"
    else
        echo "Invalid selection"
        exit 1
    fi
elif [ -n "$choice" ]; then
    if [ -d "$choice" ]; then
        SELECTED="$choice"
    else
        echo "Path does not exist"
        exit 1
    fi
fi

cd "$SELECTED" 2>/dev/null || { echo "Cannot access folder"; exit 1; }

if [ -f "audiobook_ai/core/epub_parser.py" ]; then
    echo ""
    echo "✅ Selected: $(pwd)"
    echo "   (AIGUIBook project)"
else
    echo "⚠️  Warning: Not a standard AIGUIBook project"
    echo "   Path: $(pwd)"
fi

pwd
