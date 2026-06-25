#!/bin/bash
# download_real_data.sh
# Robust downloader for BurnTrack real dataset

API_KEY="bb880143fa048ebb2da6d6d0057ae5ef"
TOTAL_DAYS=90
CHUNK_DAYS=5
CHUNKS_DIR="data/processed/chunks"
FINAL_OUTPUT="data/processed/real_african_dataset_large.csv"

mkdir -p "$CHUNKS_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Array to keep track of failed chunks
FAILED_CHUNKS=()

echo "====================================================="
echo "🔥 BurnTrack Robust Data Downloader (NASA FIRMS)"
echo "====================================================="
echo "Downloading $TOTAL_DAYS days of data in $CHUNK_DAYS-day chunks..."

# 1. Main Pass
for (( i=$TOTAL_DAYS; i>0; i-=$CHUNK_DAYS )); do
    # Calculate the start date for this chunk
    # Compatible with Linux 'date' command
    START_DATE=$(date -d "$i days ago" +%Y-%m-%d)
    CHUNK_FILE="$CHUNKS_DIR/chunk_${START_DATE}.csv"
    
    echo "-----------------------------------------------------"
    echo "📅 Processing chunk starting: $START_DATE (for $CHUNK_DAYS days)"
    
    if [ -f "$CHUNK_FILE" ]; then
        # Check if file has more than just the header (usually 1 line)
        LINES=$(wc -l < "$CHUNK_FILE")
        if [ "$LINES" -gt 1 ]; then
            echo "✅ Chunk already exists and has data ($LINES lines). Skipping."
            continue
        else
            echo "⚠️ Chunk exists but is empty (only header). Redownloading."
            rm "$CHUNK_FILE"
        fi
    fi
    
    # Run the python script
    export NASA_FIRMS_API_KEY="$API_KEY"
    python3 scripts/build_real_dataset.py --start-date "$START_DATE" --days "$CHUNK_DAYS" --output "$CHUNK_FILE"
    
    # Check if successful
    if [ $? -eq 0 ] && [ -f "$CHUNK_FILE" ]; then
        echo "✅ Chunk $START_DATE completed successfully!"
    else
        echo "❌ Chunk $START_DATE failed. Will retry later."
        FAILED_CHUNKS+=("$START_DATE")
        # Remove partial/corrupted file if it exists
        [ -f "$CHUNK_FILE" ] && rm "$CHUNK_FILE"
    fi
done

# 2. Retry Pass
if [ ${#FAILED_CHUNKS[@]} -gt 0 ]; then
    echo "====================================================="
    echo "🔄 Retrying Failed Chunks"
    echo "====================================================="
    for START_DATE in "${FAILED_CHUNKS[@]}"; do
        CHUNK_FILE="$CHUNKS_DIR/chunk_${START_DATE}.csv"
        echo "📅 Retrying chunk: $START_DATE"
        
        export NASA_FIRMS_API_KEY="$API_KEY"
        python3 scripts/build_real_dataset.py --start-date "$START_DATE" --days "$CHUNK_DAYS" --output "$CHUNK_FILE"
        
        if [ $? -eq 0 ] && [ -f "$CHUNK_FILE" ]; then
            echo "✅ Retry for $START_DATE successful!"
        else
            echo "❌ Retry for $START_DATE failed again. Skipping permanently."
        fi
    done
fi

# 3. Merge All Chunks
echo "====================================================="
echo "🔗 Merging Chunks into Final Dataset"
echo "====================================================="
if ls $CHUNKS_DIR/chunk_*.csv 1> /dev/null 2>&1; then
    # Grab the header from the first available chunk
    FIRST_FILE=$(ls $CHUNKS_DIR/chunk_*.csv | head -n 1)
    head -n 1 "$FIRST_FILE" > "$FINAL_OUTPUT"
    
    # Append the data (tail -n +2 skips the header) from all chunks
    for file in $CHUNKS_DIR/chunk_*.csv; do
        LINES=$(wc -l < "$file")
        if [ "$LINES" -gt 1 ]; then
            tail -n +2 "$file" >> "$FINAL_OUTPUT"
        fi
    done
    
    TOTAL_ROWS=$(wc -l < "$FINAL_OUTPUT")
    echo "🎉 Done! Final dataset created at: $FINAL_OUTPUT"
    echo "📊 Total rows (including header): $TOTAL_ROWS"
else
    echo "⚠️ No chunks found to merge."
fi
