function setupPreview(fileInputId, previewId, textId) {
    document.getElementById(fileInputId).addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById(previewId);
                img.src = e.target.result;
                img.style.display = 'block';
                document.getElementById(textId).style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });
}

setupPreview('file-input-1', 'preview-1', 'input-text-1');
setupPreview('file-input-2', 'preview-2', 'input-text-2');