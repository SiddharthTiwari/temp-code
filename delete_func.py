file_registry_path = "file_registry.json"

def calculate_file_hash(file_path):
    """Calculate MD5 hash for a file to use as unique identifier."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_file_registry():
    """Load the file registry that maps file hashes to file paths."""
    if os.path.exists(file_registry_path):
        with open(file_registry_path, 'r') as f:
            return json.load(f)
    return {}

def save_file_registry(registry):
    """Save the file registry to disk."""
    with open(file_registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

def delete_documents_by_file_key(vectorstore, file_key):
    """Efficiently delete documents from FAISS index by file_key."""
    try:
        all_indices = list(range(vectorstore.index.ntotal))
        
        indices_to_remove = []
        for idx in all_indices:
            doc_id = vectorstore.index_to_docstore_id[idx]
            if vectorstore.docstore._dict[doc_id].metadata.get('file_key') == file_key:
                indices_to_remove.append(idx)
        
        if not indices_to_remove:
            print(f"No documents found with file_key: {file_key}")
            return False
        
        indices_to_remove = np.sort(np.array(indices_to_remove))[::-1]
        
        for idx in indices_to_remove:
            vectorstore.index.remove_ids(np.array([idx]))
            doc_id = vectorstore.index_to_docstore_id.pop(idx)
            vectorstore.docstore._dict.pop(doc_id)
            
            for k in list(vectorstore.index_to_docstore_id.keys()):
                if k > idx:
                    vectorstore.index_to_docstore_id[k-1] = vectorstore.index_to_docstore_id.pop(k)
        
        vectorstore.save_local(index_path)
        return True
        
    except Exception as e:
        print(f"Error during deletion from index: {str(e)}")
        return False

def delete_file_by_key(file_key):
    """Delete a file using its unique key and remove it from the index."""
    registry = load_file_registry()
    
    if file_key in registry:
        file_path = registry[file_key]["path"]
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        
        registry.pop(file_key)
        save_file_registry(registry)
        
        global vectorstore_faiss
        if os.path.exists(index_path):
            success = delete_documents_by_file_key(vectorstore_faiss, file_key)
            if success:
                print(f"Updated index: Removed documents with file_key: {file_key}")
            else:
                print(f"No changes made to the index for file_key: {file_key}")
        
        return True
    else:
        print(f"No file found with key: {file_key}")
        return False




class DeleteFileRequest(BaseModel):
    file_key: str
    category: str

@app.post('/delete-file')
async def delete_file(request: DeleteFileRequest):
    try:
        registry = load_file_registry()
        
        if request.file_key not in registry:
            return JSONResponse(
                status_code=404,
                content={'error': f"No file found with key: {request.file_key}"}
            )
        
        file_info = registry[request.file_key]
        
        if file_info.get("category") != request.category:
            return JSONResponse(
                status_code=403,
                content={'error': f"File does not belong to category: {request.category}"}
            )
            
        result = delete_file_by_key(request.file_key)
        
        if result:
            return JSONResponse(content={'response': 'File deleted successfully'})
        else:
            return JSONResponse(
                status_code=500,
                content={'error': 'Failed to delete the file'}
            )
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )
