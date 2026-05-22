from typing import List, Dict, Any

class RecursiveTextSplitter:
    """
    Splits text recursively based on a list of separators (e.g. \n\n, \n, space, char)
    so chunks fit within chunk_size while maintaining semantic integrity.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Core recursive splitting logic."""
        final_chunks = []
        
        # Determine the separator to use
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
                
        # Split text by separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)
            
        # Recombine splits into reasonably sized chunks
        good_splits = []
        for s in splits:
            if separator != "":
                # Don't strip too early to avoid losing spacer information, but let's be clean
                good_splits.append(s)
            else:
                good_splits.append(s)
                
        # Buffer to build up chunks
        current_chunk = []
        current_len = 0
        
        for split in good_splits:
            split_len = len(split)
            
            # If a single split exceeds chunk_size, we split it recursively
            if split_len > self.chunk_size:
                # Flush current buffer first
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Split this oversized piece recursively using deeper separators
                if new_separators:
                    recursive_splits = self._split_text(split, new_separators)
                    final_chunks.extend(recursive_splits)
                else:
                    # No separators left, just slice it up by characters
                    i = 0
                    while i < split_len:
                        final_chunks.append(split[i:i + self.chunk_size])
                        i += self.chunk_size - self.chunk_overlap
            else:
                # Check if adding this split exceeds chunk_size
                # Plus length of separator if not empty and already have items
                sep_len = len(separator) if current_chunk else 0
                if current_len + split_len + sep_len > self.chunk_size:
                    # Flush current buffer
                    if current_chunk:
                        final_chunks.append(separator.join(current_chunk))
                        
                    # Calculate overlap: keep last N splits that fit within chunk_overlap
                    overlap_splits = []
                    overlap_len = 0
                    for prev_split in reversed(current_chunk):
                        prev_sep_len = len(separator) if overlap_splits else 0
                        if overlap_len + len(prev_split) + prev_sep_len <= self.chunk_overlap:
                            overlap_splits.insert(0, prev_split)
                            overlap_len += len(prev_split) + prev_sep_len
                        else:
                            break
                    
                    current_chunk = overlap_splits
                    current_len = overlap_len
                
                current_chunk.append(split)
                current_len += split_len + (len(separator) if len(current_chunk) > 1 else 0)
                
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))
            
        # Strip output chunks for neatness
        return [c.strip() for c in final_chunks if c.strip()]

    def split_document(self, text: str, document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits text and associates metadata with each chunk.
        """
        raw_chunks = self._split_text(text, self.separators)
        chunks_with_metadata = []
        
        for idx, chunk_text in enumerate(raw_chunks):
            # Shallow copy the original metadata
            chunk_meta = document_metadata.copy()
            chunk_meta["chunk_index"] = idx
            
            # Simple heuristic to guess page number for PDFs based on "--- Page Break ---"
            if "pages" in document_metadata and document_metadata["file_type"] == "pdf":
                # Find how many page breaks appear before this text chunk
                # (approximate by looking at character indices in the full text)
                pass # metadata is already loaded; we keep it clean.
                
            chunks_with_metadata.append({
                "text": chunk_text,
                "metadata": chunk_meta
            })
            
        return chunks_with_metadata
