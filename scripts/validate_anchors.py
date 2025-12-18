#!/usr/bin/env python3
"""
Simple anchor link validator script.
Validates that all anchor links in markdown files point to valid headings.
"""

import re
from pathlib import Path


def heading_to_anchor(heading: str) -> str:
    """Convert a heading text to its anchor format (GitHub/CommonMark style)."""
    anchor = heading.lower()
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    anchor = re.sub(r'\s+', '-', anchor)
    anchor = anchor.strip('-')
    return anchor


def extract_headings(content: str) -> list:
    """Extract all headings from markdown content and convert to anchors."""
    heading_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
    headings = []
    for match in heading_pattern.finditer(content):
        heading_text = match.group(1).strip()
        anchor = heading_to_anchor(heading_text)
        headings.append(anchor)
    return headings


def extract_anchor_links(content: str) -> list:
    """Extract all anchor links from markdown content."""
    # Pattern for [text](#anchor)
    link_pattern = re.compile(r'\[([^\]]*)\]\(#([^)]+)\)')
    links = []
    for line_num, line in enumerate(content.split('\n'), start=1):
        for match in link_pattern.finditer(line):
            link_text = match.group(1)
            anchor = match.group(2)
            links.append((line_num, link_text, anchor))
    return links


def validate_file(file_path: Path) -> tuple:
    """Validate anchor links in a single file."""
    content = file_path.read_text(encoding='utf-8')
    headings = extract_headings(content)
    anchor_links = extract_anchor_links(content)
    
    valid = []
    broken = []
    
    for line_num, link_text, anchor in anchor_links:
        if anchor in headings:
            valid.append((line_num, link_text, anchor))
        else:
            broken.append((line_num, link_text, anchor, headings))
    
    return valid, broken, headings


def find_all_markdown_files(root: Path = Path('.')) -> list:
    """Find all markdown files in the project, excluding certain directories."""
    excluded_dirs = {'.venv', 'node_modules', '.git', '__pycache__', 'dist', 'build', '.hypothesis', '.kiro', 'htmlcov', '.pytest_cache'}
    
    files = []
    for md_file in root.glob('**/*.md'):
        if not any(excluded in md_file.parts for excluded in excluded_dirs):
            files.append(md_file)
    return sorted(files)


def main():
    # Find all markdown files in project
    files = find_all_markdown_files()
    
    total_valid = 0
    total_broken = 0
    all_broken = []
    
    for file_path in files:
        if not file_path.exists():
            print(f"SKIP: {file_path} (not found)")
            continue
            
        valid, broken, headings = validate_file(file_path)
        total_valid += len(valid)
        total_broken += len(broken)
        
        if broken:
            print(f"\n{file_path}: {len(broken)} broken anchor(s)")
            for line_num, link_text, anchor, available in broken:
                print(f"  Line {line_num}: [{link_text}](#{anchor})")
                # Find similar headings
                similar = [h for h in available if anchor in h or h in anchor]
                if similar:
                    print(f"    Similar: {similar[:3]}")
                all_broken.append((file_path, line_num, link_text, anchor))
        else:
            print(f"{file_path}: {len(valid)} anchor(s) - all valid ✓")
    
    print(f"\n=== Summary ===")
    print(f"Total anchor links: {total_valid + total_broken}")
    print(f"Valid: {total_valid}")
    print(f"Broken: {total_broken}")
    
    if all_broken:
        print(f"\n=== Broken Links ===")
        for file_path, line_num, link_text, anchor in all_broken:
            print(f"  {file_path}:{line_num} - #{anchor}")


if __name__ == "__main__":
    main()
