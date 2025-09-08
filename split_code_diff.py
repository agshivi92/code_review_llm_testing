import subprocess
import tiktoken
from whatthepatch import parse_patch
import re

# --- Configuration ---
# The target context window size for the LLM.
CONTEXT_WINDOW_TOKENS = 128000 
# The base branch to compare against for generating the diff.
#BASE_BRANCH = "origin/main" 

def count_tokens(text: str, tokenizer) -> int:
    """Calculates the number of tokens in a given text."""
    return len(tokenizer.encode(text))

def get_git_diff(base_branch: str) -> str:
    """Generates the git diff against the specified base branch."""
    try:
        # Using unified=0 removes all context lines, creating the smallest possible diff.
        args = ["git", "diff", "--unified=0", base_branch]
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e.stderr}")
        return ""

def format_patch_from_hunks(patch, hunks) -> str:
    """Reconstructs a diff string from a patch header and a list of hunks."""
    header = f"diff --git a/{patch.header.old_path} b/{patch.header.new_path}\n"
    header += f"--- a/{patch.header.old_path}\n"
    header += f"+++ b/{patch.header.new_path}\n"
    
    hunk_texts = []
    for hunk in hunks:
        hunk_texts.append(str(hunk))
        
    return header + "\n".join(hunk_texts)

def split_file_diff(patch, tokenizer) -> list[str]:
    """
    Splits a single file's diff into chunks based on functions/classes.
    This is the fallback when a whole file's diff is too large.
    """
    chunks = []
    current_hunks = []
    current_chunk_tokens = 0
    
    # Estimate header tokens once; it's part of every chunk from this file.
    #header_str = f"diff --git a/{patch.header.old_path} b/{patch.header.new_path}\n--- a/{patch.header.old_path}\n+++ b/{patch.header.new_path}\n"
    header_tokens = count_tokens(code_to_review, tokenizer)

    for hunk in patch.hunks:
        hunk_text = str(hunk)
        hunk_tokens = count_tokens(hunk_text, tokenizer)

        # Check for function/class definition in the hunk header (the '@@' line)
        # This signals a good logical point to split the diff.
        is_new_logical_block = hunk.section_header and re.match(r'^(class|def)\s+', hunk.section_header.strip())

        # If we find a new block and the current chunk is not empty,
        # we finalize the previous chunk.
        if is_new_logical_block and current_hunks:
            chunk_diff = format_patch_from_hunks(patch, current_hunks)
            chunks.append(chunk_diff)
            current_hunks = []
            current_chunk_tokens = 0
        
        # Add the new hunk to the current chunk.
        current_hunks.append(hunk)
        current_chunk_tokens += hunk_tokens

        # If even a single function/class chunk is getting too big, split it by hunk.
        # This is the final fallback to ensure we never breach the context window.
        if header_tokens + current_chunk_tokens > CONTEXT_WINDOW_TOKENS:
            # Pop the last hunk that made it overflow
            overflow_hunk = current_hunks.pop()
            
            # Finalize the chunk without the overflowing hunk (if there's anything left)
            if current_hunks:
                chunk_diff = format_patch_from_hunks(patch, current_hunks)
                chunks.append(chunk_diff)
            
            # The overflow hunk becomes the start of the next chunk.
            current_hunks = [overflow_hunk]
            current_chunk_tokens = count_tokens(str(overflow_hunk), tokenizer)

    # Add the last remaining chunk
    if current_hunks:
        chunk_diff = format_patch_from_hunks(patch, current_hunks)
        chunks.append(chunk_diff)
        
    return chunks


def create_diff_chunks(code_to_review) -> list[str]:
    """
    Main function to generate git diff and split it into chunks
    that fit within the LLM's context window.
    """
    print("Initializing tokenizer...")
    tokenizer = tiktoken.get_encoding("cl100k_base")

    #print(f"Generating diff against '{BASE_BRANCH}'...")
    #full_diff = get_git_diff(BASE_BRANCH)
    full_diff = code_to_review

    # --- New: Define a directory to save your chunks ---
    # This directory will be created in the GitHub Actions runner's workspace
    OUTPUT_CHUNKS_DIR = "_temp_diff_chunks"
    os.makedirs(OUTPUT_CHUNKS_DIR, exist_ok=True) # Ensure the directory exists
    if not full_diff:
        print("No diff found or git error occurred.")
        return []

    # --- Level 1: Try the entire diff first ---
    total_tokens = count_tokens(full_diff, tokenizer)
    print(f"Total diff has {total_tokens} tokens.")
    if total_tokens <= CONTEXT_WINDOW_TOKENS:
        print("Entire diff fits within the context window. Creating one chunk.")
        chunk_filename = os.path.join(OUTPUT_CHUNKS_DIR, "full_diff_chunk_0.diff")
        with open(chunk_filename, "w") as f:
            f.write(full_diff)
        print(f" -> Saved full diff chunk to {chunk_filename}")
        final_chunks = [full_diff]
        return final_chunks

    # --- Level 2: Diff is too large, split by file ---
    print("Total diff exceeds context window. Splitting by file...")
    patches = list(parse_patch(full_diff))
    final_chunks = []
    chunk_counter = 0
    for patch in patches:
        # We only care about Python files for this logic
        if not patch.header.new_path.endswith(".py"):
            continue

        file_diff_str = str(patch)
        file_tokens = count_tokens(file_diff_str, tokenizer)
        
        print(f"\nProcessing file: {patch.header.new_path} ({file_tokens} tokens)")

        if file_tokens <= CONTEXT_WINDOW_TOKENS:
            print(" -> Fits in context window. Adding as a single chunk.")
            final_chunks.append(file_diff_str)
            # --- NEW: Save this chunk to a file ---
            # Use a unique name for each chunk
            chunk_filename = os.path.join(OUTPUT_CHUNKS_DIR, f"diff_chunk_{chunk_counter}.diff")
            with open(chunk_filename, "w") as f:
                f.write(file_diff_str)
            print(f" -> Saved chunk to {chunk_filename}")
            chunk_counter += 1
        else:
            # --- Level 3: File is too large, split by function/class ---
            print(" -> File diff is too large. Splitting by function/class...")
            file_chunks = split_file_diff(patch, tokenizer)
            print(f" -> Split into {len(file_chunks)} smaller chunks.")
            final_chunks.extend(file_chunks)
            chunk_filename = os.path.join(OUTPUT_CHUNKS_DIR, f"diff_chunk_oversized_{chunk_counter}.diff")
            with open(chunk_filename, "w") as f:
                f.write(file_diff_str)
            print(f" -> Saved oversized chunk to {chunk_filename}")
            chunk_counter += 1
    
    return final_chunks

if __name__ == "__main__":
    print(f"Reading diff from file diff_code_to_review")
    with open('diff_code_to_review.txt', 'r') as file:
      code_to_review = file.read()
    diff_chunks = create_diff_chunks(code_to_review)

    print("\n" + "="*50)
    print(f"Generated {len(diff_chunks)} chunk(s) for the LLM.")
    print("="*50 + "\n")

    for i, chunk in enumerate(diff_chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk[:500] + "\n..." if len(chunk) > 500 else chunk) # Print a preview
        print("-"*(len(f"--- Chunk {i+1} ---")) + "\n")
