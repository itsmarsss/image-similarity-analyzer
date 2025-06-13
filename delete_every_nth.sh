#!/bin/bash
"""
delete_every_nth.sh

Delete every nth file from a directory based on alphabetical/numerical sorting.
Useful for reducing the number of video frames or similar sequential files.

Usage:
    ./delete_every_nth.sh <directory> <interval> [options]
    
Arguments:
    directory    Directory containing files to process
    interval     Delete every nth file (e.g., 2 = delete every 2nd file)
    
Options:
    --pattern PATTERN    File pattern to match (default: *.png)
    --inverse           Keep every nth file instead of deleting every nth file
    --dry-run           Show what would be deleted without actually deleting
    --force             Skip confirmation prompt
    --verbose           Show detailed output
    --help              Show this help message

Examples:
    ./delete_every_nth.sh ./cropped_faces 2                    # Delete every 2nd PNG file
    ./delete_every_nth.sh ./cropped_faces 2 --inverse          # Keep every 2nd file, delete the rest
    ./delete_every_nth.sh ./cropped_faces 3 --dry-run          # Preview deletion of every 3rd file
    ./delete_every_nth.sh ./cropped_faces 5 --pattern "*.jpg"  # Delete every 5th JPG file
    ./delete_every_nth.sh ./cropped_faces 2 --force --verbose  # Delete every 2nd file without confirmation
"""

set -e  # Exit on any error

# Default values
PATTERN="*.png"
INVERSE=false
DRY_RUN=false
FORCE=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show help
show_help() {
    echo "Delete every nth file from a directory"
    echo ""
    echo "Usage: $0 <directory> <interval> [options]"
    echo ""
    echo "Arguments:"
    echo "  directory    Directory containing files to process"
    echo "  interval     Delete every nth file (e.g., 2 = delete every 2nd file)"
    echo ""
    echo "Options:"
    echo "  --pattern PATTERN    File pattern to match (default: *.png)"
    echo "  --inverse           Keep every nth file instead of deleting every nth file"
    echo "  --dry-run           Show what would be deleted without actually deleting"
    echo "  --force             Skip confirmation prompt"
    echo "  --verbose           Show detailed output"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 ./cropped_faces 2                    # Delete every 2nd PNG file"
    echo "  $0 ./cropped_faces 2 --inverse          # Keep every 2nd file, delete the rest"
    echo "  $0 ./cropped_faces 3 --dry-run          # Preview deletion of every 3rd file"
    echo "  $0 ./cropped_faces 5 --pattern \"*.jpg\"  # Delete every 5th JPG file"
    echo "  $0 ./cropped_faces 2 --force --verbose  # Delete every 2nd file without confirmation"
}

# Function to log messages
log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

# Function to log warnings
warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Function to log errors
error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to log success
success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Parse command line arguments
DIRECTORY=""
INTERVAL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --inverse)
            INVERSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
        *)
            if [ -z "$DIRECTORY" ]; then
                DIRECTORY="$1"
            elif [ -z "$INTERVAL" ]; then
                INTERVAL="$1"
            else
                error "Too many arguments"
                echo "Use --help for usage information."
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$DIRECTORY" ] || [ -z "$INTERVAL" ]; then
    error "Missing required arguments"
    echo "Usage: $0 <directory> <interval> [options]"
    echo "Use --help for more information."
    exit 1
fi

# Validate directory exists
if [ ! -d "$DIRECTORY" ]; then
    error "Directory '$DIRECTORY' does not exist"
    exit 1
fi

# Validate interval is a positive integer
if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    error "Interval must be a positive integer (got: '$INTERVAL')"
    exit 1
fi

if [ "$INTERVAL" -eq 1 ]; then
    error "Interval of 1 would delete ALL files. Use a value > 1."
    exit 1
fi

# Get absolute path
DIRECTORY=$(cd "$DIRECTORY" && pwd)

log "Processing directory: $DIRECTORY"
log "File pattern: $PATTERN"
if [ "$INVERSE" = true ]; then
    log "Mode: Keep every ${INTERVAL} files (delete the rest)"
else
    log "Mode: Delete every ${INTERVAL} files"
fi
log "Dry run mode: $DRY_RUN"

# Get list of files matching pattern, sorted
FILES=()
while IFS= read -r -d '' file; do
    FILES+=("$file")
done < <(find "$DIRECTORY" -maxdepth 1 -name "$PATTERN" -type f -print0 | sort -z)

TOTAL_FILES=${#FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    warn "No files found matching pattern '$PATTERN' in directory '$DIRECTORY'"
    exit 0
fi

log "Found $TOTAL_FILES files matching pattern"

# Calculate which files will be deleted
TO_DELETE=()
if [ "$INVERSE" = true ]; then
    # Keep every nth file, delete the rest
    for ((i=0; i<TOTAL_FILES; i++)); do
        if [ $((i % INTERVAL)) -ne $((INTERVAL-1)) ]; then
            TO_DELETE+=("${FILES[$i]}")
        fi
    done
else
    # Delete every nth file (original behavior)
    for ((i=INTERVAL-1; i<TOTAL_FILES; i+=INTERVAL)); do
        TO_DELETE+=("${FILES[$i]}")
    done
fi

DELETE_COUNT=${#TO_DELETE[@]}

if [ "$DELETE_COUNT" -eq 0 ]; then
    warn "No files to delete (interval $INTERVAL is larger than file count $TOTAL_FILES)"
    exit 0
fi

# Show what will be deleted
echo ""
echo "=== DELETION SUMMARY ==="
echo "Directory: $DIRECTORY"
echo "Pattern: $PATTERN"
echo "Total files found: $TOTAL_FILES"
if [ "$INVERSE" = true ]; then
    echo "Mode: Keep every ${INTERVAL} files (delete the rest)"
    echo "Files to keep: $((TOTAL_FILES - DELETE_COUNT))"
    echo "Files to delete: $DELETE_COUNT"
else
    echo "Mode: Delete every ${INTERVAL} files"
    echo "Files to delete: $DELETE_COUNT"
    echo "Files remaining after deletion: $((TOTAL_FILES - DELETE_COUNT))"
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "=== FILES THAT WOULD BE DELETED (DRY RUN) ==="
    for file in "${TO_DELETE[@]}"; do
        echo "  $(basename "$file")"
    done
    echo ""
    success "Dry run completed. No files were actually deleted."
    exit 0
fi

echo ""
echo "=== FILES TO BE DELETED ==="
for file in "${TO_DELETE[@]}"; do
    echo "  $(basename "$file")"
done

# Confirmation prompt (unless --force is used)
if [ "$FORCE" != true ]; then
    echo ""
    echo -e "${YELLOW}WARNING: This will permanently delete $DELETE_COUNT files!${NC}"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi
fi

# Perform deletion
echo ""
echo "=== DELETING FILES ==="
DELETED_COUNT=0
FAILED_COUNT=0

for file in "${TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        if rm "$file" 2>/dev/null; then
            DELETED_COUNT=$((DELETED_COUNT + 1))
            log "Deleted: $(basename "$file")"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            error "Failed to delete: $(basename "$file")"
        fi
    else
        warn "File no longer exists: $(basename "$file")"
    fi
done

# Final summary
echo ""
echo "=== DELETION COMPLETED ==="
echo "Files successfully deleted: $DELETED_COUNT"
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "Files failed to delete: $FAILED_COUNT"
fi
echo "Files remaining in directory: $((TOTAL_FILES - DELETED_COUNT))"

if [ "$DELETED_COUNT" -gt 0 ]; then
    success "Successfully deleted $DELETED_COUNT files"
else
    warn "No files were deleted"
fi 