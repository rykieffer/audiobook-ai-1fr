#!/bin/bash
# Interactive project folder selection with mouse support

echo "============================================"
echo "📁 AIGUIBook Project Folder Selection"
echo "============================================"
echo ""
echo "Available folders in current directory:"
echo ""

# List folders with numbers
FOLDERS=()
COUNTER=1
select item in */; do
    if [ -d "$item" ]; then
        folder_name="${item%/}"
        if [[ ! "$folder_name" == .* ]]; then
            FOLDERS+=("$folder_name")
            echo "  $COUNTER) $folder_name"
            ((COUNTER++))
        fi
    fi
done

echo ""
echo "Or enter the folder path directly"
echo ""

# Use select for menu-driven choice
PS3="Please choose a folder (1-${#FOLDERS[@]}) or enter path: "
select choice in "${FOLDERS[@]}" "Custom Path"; do
    if [ -n "$choice" ]; then
        if [ "$REPLY" = "Custom Path" ] || [ "$REPLY" -gt "${#FOLDERS[@]}" ] 2>/dev/null; then
            read -p "Enter folder path: " custom_path
            if [ -d "$custom_path" ]; then
                SELECTED="$custom_path"
                break
            else
                echo "Invalid path, try again"
            fi
        else
            SELECTED="${FOLDERS[$((REPLY-1))]}"
            break
        fi
    fi
done

if [ -f "$SELECTED/audiobook_ai/core/epub_parser.py" ]; then
    echo ""
    echo "✅ Selected folder: $SELECTED"
    echo "   (AIGUIBook project detected)"
else
    echo ""
    echo "⚠️  Warning: May not be an AIGUIBook project"
    echo "   Selected: $SELECTED"
fi

echo "$SELECTED"
