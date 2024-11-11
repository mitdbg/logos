"""Generate the code reference pages and navigation.

Script was taken from
https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages
"""
"""Generate the code reference pages and navigation."""

"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files


# Replace the file docs/index.md with a copy of README.md
with mkdocs_gen_files.open("index.md", "w") as fd:
    with open("README.md") as readme:
        fd.write(readme.read())


nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src" 

for path in sorted(src.rglob("logos*/*.py")):  
    module_path = path.relative_to(src).with_suffix("")  
    doc_path = path.relative_to(src).with_suffix(".md")  
    full_doc_path = Path("reference", doc_path)  

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":  
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    
    print(doc_path.as_posix())

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  
        identifier = ".".join(parts)  
        print("::: " + identifier, file=fd)  

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  
    nav_file.writelines(nav.build_literate_nav()) 