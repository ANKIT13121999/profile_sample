import requests
import base64
from typing import List, Dict
import os

class GitHubDDLReader:
    """
    Reads DDL files from a GitHub repository using GitHub API.
    Works in both local and deployed environments.
    """
    
    def __init__(self, token: str = None):
        """
        Initialize GitHub reader.
        
        Args:
            token: GitHub personal access token (optional for public repos)
        """
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
    
    def get_folder_contents(self, owner: str, repo: str, folder_path: str, 
                           branch: str = 'main') -> List[Dict]:
        """
        Get all files from a specific folder in GitHub repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            folder_path: Path to folder (e.g., 'ddl' or 'database/ddl')
            branch: Branch name (default: 'main')
            
        Returns:
            List of file metadata dictionaries
        """
        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{folder_path}'
        params = {'ref': branch}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_file_content(self, owner: str, repo: str, file_path: str, 
                        branch: str = 'main') -> str:
        """
        Get content of a specific file from GitHub.
        
        Args:
            owner: Repository owner
            repo: Repository name
            file_path: Full path to file
            branch: Branch name
            
        Returns:
            File content as string
        """
        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{file_path}'
        params = {'ref': branch}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        content_data = response.json()
        
        # Decode base64 content
        content = base64.b64decode(content_data['content']).decode('utf-8')
        return content
    
    def get_all_ddl_files(self, owner: str, repo: str, folder_path: str, 
                         branch: str = 'main', file_extensions: List[str] = None) -> Dict[str, str]:
        """
        Get all DDL files from a folder.
        
        Args:
            owner: Repository owner
            repo: Repository name
            folder_path: Path to DDL folder
            branch: Branch name
            file_extensions: List of file extensions to filter (e.g., ['.sql', '.ddl'])
            
        Returns:
            Dictionary with filename as key and content as value
        """
        if file_extensions is None:
            file_extensions = ['.sql', '.ddl']
        
        ddl_files = {}
        
        # Get folder contents
        contents = self.get_folder_contents(owner, repo, folder_path, branch)
        
        for item in contents:
            # Filter by file type and extension
            if item['type'] == 'file':
                file_name = item['name']
                if any(file_name.endswith(ext) for ext in file_extensions):
                    try:
                        content = self.get_file_content(owner, repo, item['path'], branch)
                        ddl_files[file_name] = content
                        print(f"✓ Loaded: {file_name}")
                    except Exception as e:
                        print(f"✗ Error loading {file_name}: {str(e)}")
        
        return ddl_files
    
    def get_ddl_files_recursive(self, owner: str, repo: str, folder_path: str, 
                               branch: str = 'main', file_extensions: List[str] = None) -> Dict[str, str]:
        """
        Recursively get all DDL files from a folder and its subfolders.
        
        Args:
            owner: Repository owner
            repo: Repository name
            folder_path: Path to DDL folder
            branch: Branch name
            file_extensions: List of file extensions to filter
            
        Returns:
            Dictionary with relative file path as key and content as value
        """
        if file_extensions is None:
            file_extensions = ['.sql', '.ddl']
        
        ddl_files = {}
        
        def process_folder(path: str):
            contents = self.get_folder_contents(owner, repo, path, branch)
            
            for item in contents:
                if item['type'] == 'file':
                    file_name = item['name']
                    if any(file_name.endswith(ext) for ext in file_extensions):
                        try:
                            content = self.get_file_content(owner, repo, item['path'], branch)
                            ddl_files[item['path']] = content
                            print(f"✓ Loaded: {item['path']}")
                        except Exception as e:
                            print(f"✗ Error loading {item['path']}: {str(e)}")
                elif item['type'] == 'dir':
                    # Recursively process subdirectories
                    process_folder(item['path'])
        
        process_folder(folder_path)
        return ddl_files


# Usage Example
def main():
    # Initialize reader
    # For private repos, set GITHUB_TOKEN environment variable or pass token
    reader = GitHubDDLReader()
    
    # Configure your repository details
    OWNER = 'your-github-username'
    REPO = 'your-repo-name'
    DDL_FOLDER = 'ddl'  # or 'database/ddl' or whatever your folder structure is
    BRANCH = 'main'  # or 'master', 'develop', etc.
    
    try:
        # Get all DDL files from the folder
        print(f"Reading DDL files from {OWNER}/{REPO}/{DDL_FOLDER}...")
        ddl_files = reader.get_all_ddl_files(
            owner=OWNER,
            repo=REPO,
            folder_path=DDL_FOLDER,
            branch=BRANCH,
            file_extensions=['.sql', '.ddl']
        )
        
        print(f"\n✓ Successfully loaded {len(ddl_files)} DDL files")
        
        # Process the DDL files
        for filename, content in ddl_files.items():
            print(f"\n{'='*50}")
            print(f"File: {filename}")
            print(f"{'='*50}")
            print(content[:200] + "..." if len(content) > 200 else content)
            
            # Here you can:
            # - Execute DDLs against database
            # - Parse and validate DDLs
            # - Transform DDLs
            # - Store in database catalog
            
    except requests.exceptions.HTTPError as e:
        print(f"Error accessing GitHub: {e}")
        print("Make sure you have:")
        print("1. Correct repository details")
        print("2. GitHub token (for private repos)")
        print("3. Proper permissions")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
