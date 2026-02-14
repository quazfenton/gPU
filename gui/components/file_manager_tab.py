"""
File Manager Tab component for GUI interface.

This module provides the UI component for uploading and managing files
that can be used as inputs for ML jobs.
"""

from typing import Tuple
import gradio as gr
import pandas as pd

from gui.components.file_upload_handler import FileUploadHandler
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class FileManagerTab(LoggerMixin):
    """UI component for file upload and management."""
    
    def __init__(self, file_handler: FileUploadHandler):
        """Initialize file manager tab."""
        self.file_handler = file_handler
        self.logger.info("FileManagerTab initialized")
    
    def render(self) -> gr.Blocks:
        """Render the file manager tab."""
        with gr.Blocks() as tab:
            gr.Markdown("## File Manager")
            gr.Markdown("Upload files to use as inputs for ML jobs.")
            
            # Upload section
            gr.Markdown("### Upload Files")
            
            with gr.Row():
                file_upload = gr.File(
                    label="Select File to Upload",
                    file_count="single",
                    type="filepath"
                )
                upload_btn = gr.Button("📤 Upload", variant="primary")
            
            upload_status = gr.Markdown(value="", visible=False)
            
            gr.Markdown("---")
            
            # Uploaded files list
            gr.Markdown("### Uploaded Files")
            
            refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
            
            files_df = gr.Dataframe(
                label="Your Uploaded Files",
                headers=["Filename", "Size (KB)", "Uploaded", "File Path"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )
            
            # File actions
            with gr.Row():
                selected_file = gr.Textbox(
                    label="Selected File",
                    placeholder="Click on a file in the table above",
                    interactive=True
                )
                delete_btn = gr.Button("🗑️ Delete", variant="stop")
            
            action_status = gr.Markdown(value="", visible=False)
            
            # Usage instructions
            gr.Markdown("""
### How to Use Uploaded Files

1. **Upload a file** using the upload button above
2. **Copy the file path** from the table
3. **Use the path in job submission** - paste it into the JSON input field

**Example:**
```json
{
  "audio_file": "/path/to/your/uploaded/file.mp3",
  "task": "transcribe"
}
```

**Note:** For production use, files would be uploaded to cloud storage (S3, GCS) and you would use HTTP URLs instead of file paths.
""")
            
            # Event handlers
            def handle_upload(file_path):
                if not file_path:
                    return (
                        gr.Markdown(value="❌ No file selected", visible=True),
                        self._get_files_dataframe()
                    )
                
                success, saved_path, message = self.file_handler.save_uploaded_file(file_path)
                
                if success:
                    status_msg = f"""
✅ **File Uploaded Successfully!**

**File Path:** `{saved_path}`

Copy this path and use it in your job submission JSON.
"""
                else:
                    status_msg = f"❌ **Upload Failed:** {message}"
                
                return (
                    gr.Markdown(value=status_msg, visible=True),
                    self._get_files_dataframe()
                )
            
            def handle_delete(filename):
                if not filename:
                    return (
                        gr.Markdown(value="❌ No file selected", visible=True),
                        self._get_files_dataframe()
                    )
                
                success, message = self.file_handler.delete_file(filename)
                
                if success:
                    status_msg = f"✅ {message}"
                else:
                    status_msg = f"❌ {message}"
                
                return (
                    gr.Markdown(value=status_msg, visible=True),
                    self._get_files_dataframe()
                )
            
            # Load files on tab open
            tab.load(
                fn=lambda: self._get_files_dataframe(),
                outputs=[files_df]
            )
            
            # Upload button
            upload_btn.click(
                fn=handle_upload,
                inputs=[file_upload],
                outputs=[upload_status, files_df]
            )
            
            # Refresh button
            refresh_btn.click(
                fn=lambda: self._get_files_dataframe(),
                outputs=[files_df]
            )
            
            # Delete button
            delete_btn.click(
                fn=handle_delete,
                inputs=[selected_file],
                outputs=[action_status, files_df]
            )
        
        return tab
    
    def _get_files_dataframe(self) -> pd.DataFrame:
        """Get uploaded files as DataFrame."""
        try:
            files = self.file_handler.list_uploaded_files()
            
            if not files:
                return pd.DataFrame(columns=[
                    "Filename", "Size (KB)", "Uploaded", "File Path"
                ])
            
            rows = []
            for f in files:
                size_kb = f['size'] / 1024
                rows.append([
                    f['filename'],
                    f"{size_kb:.2f}",
                    f['modified'],
                    f['path']
                ])
            
            return pd.DataFrame(rows, columns=[
                "Filename", "Size (KB)", "Uploaded", "File Path"
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to get files dataframe: {e}")
            return pd.DataFrame(columns=[
                "Filename", "Size (KB)", "Uploaded", "File Path"
            ])
