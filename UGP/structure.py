import os

def print_tree(start_path, indent=""):
    files = []
    dirs = []
    for entry in os.scandir(start_path):
        if entry.is_dir():
            dirs.append(entry.name)
        else:
            files.append(entry.name)

    # Sort alphabetically
    dirs.sort()
    files.sort()

    for d in dirs:
        print(f"{indent}├── {d}/")
        print_tree(os.path.join(start_path, d), indent + "│   ")

    for f in files:
        print(f"{indent}└── {f}")

if __name__ == "__main__":
    project_path = os.path.abspath(".")  # Current directory
    print(f"Project structure for: {project_path}\n")
    print_tree(project_path)
