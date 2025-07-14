#!/usr/bin/env python3
"""
Script to prepare a release by updating the version number in __init__.py
"""
import sys
import re
import argparse
from pathlib import Path

def update_version(new_version):
    """Update the version in __init__.py"""
    init_file = Path("src/pepe/__init__.py")
    
    if not init_file.exists():
        print(f"Error: {init_file} not found!")
        return False
    
    # Read the file
    content = init_file.read_text()
    
    # Find current version
    version_match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if not version_match:
        print("Error: Could not find __version__ in __init__.py")
        return False
    
    current_version = version_match.group(1)
    print(f"Current version: {current_version}")
    
    # Replace version
    new_content = re.sub(
        r'__version__ = ["\']([^"\']+)["\']',
        f'__version__ = "{new_version}"',
        content
    )
    
    # Write back
    init_file.write_text(new_content)
    print(f"Updated version to: {new_version}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare release by updating version")
    parser.add_argument("version", help="New version number (e.g., 1.0.0)")
    parser.add_argument("--dev", action="store_true", help="Add -dev suffix for development")
    
    args = parser.parse_args()
    
    version = args.version
    if args.dev:
        version += "-dev"
    
    if update_version(version):
        print(f"\n✅ Version updated successfully!")
        print(f"Next steps:")
        if args.dev:
            print(f"  1. git add src/pepe/__init__.py")
            print(f"  2. git commit -m 'Bump version to {version} for next development cycle'")
        else:
            print(f"  1. git add src/pepe/__init__.py")
            print(f"  2. git commit -m 'Bump version to {version} for release'")
            print(f"  3. Test thoroughly, then merge to main")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
