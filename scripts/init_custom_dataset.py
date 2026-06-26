"""
Custom Dataset Bootstrap Script
Creates a standard folder layout for new sign-language datasets (NSL, BSL, ISL, etc.).

Examples:
    python scripts/init_custom_dataset.py --root data/raw/nsl --language-code NSL --language-name "Nepali Sign Language" --preset alphabet_numbers
    python scripts/init_custom_dataset.py --root data/raw/custom --language-code CSL --classes HELLO THANK_YOU YES NO
"""

import argparse
import json
from pathlib import Path
from typing import List


PRESETS = {
    "alphabet": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "numbers": [str(i) for i in range(10)],
    "alphabet_numbers": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)],
    "starter_words": ["HELLO", "THANK_YOU", "YES", "NO", "PLEASE", "SORRY"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a custom sign-language dataset scaffold")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory to create")
    parser.add_argument("--language-code", type=str, required=True, help="Language code, e.g., NSL")
    parser.add_argument("--language-name", type=str, required=True, help="Language display name")
    parser.add_argument(
        "--preset",
        type=str,
        default="alphabet_numbers",
        choices=sorted(PRESETS.keys()),
        help="Class preset to bootstrap",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional custom class list (overrides --preset)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Recommended target number of images per class for planning",
    )
    return parser.parse_args()


def write_dataset_readme(root: Path, language_code: str, language_name: str, classes: List[str], samples_per_class: int) -> None:
    lines = [
        f"# {language_name} ({language_code.upper()}) Dataset",
        "",
        "## Structure",
        "",
        "```text",
        f"{root.name}/",
        "  classes.txt",
        "  class_mapping.json",
        "  capture_notes.md",
        "  <CLASS_1>/",
        "    img_0001.jpg",
        "    ...",
        "  <CLASS_2>/",
        "    ...",
        "```",
        "",
        "## Data Collection Rules",
        "",
        "- One dominant sign per image",
        "- Good lighting and clear hand visibility",
        "- Include background/skin-tone/camera variation",
        "- Keep class labels stable over time",
        f"- Target at least {samples_per_class} images per class",
        "",
        "## Next Step",
        "",
        "Run preprocessing:",
        "",
        "```bash",
        "python scripts/preprocess_dataset.py --input <THIS_FOLDER> --output data/processed/<language_code_lower> \\",
        f"  --language-code {language_code.upper()} --language-name \"{language_name}\" --classes-file <THIS_FOLDER>/classes.txt",
        "```",
    ]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    classes = [c.strip() for c in (args.classes or PRESETS[args.preset]) if c.strip()]
    if not classes:
        raise ValueError("No classes provided after processing --classes/--preset")

    # Create class folders
    for class_name in classes:
        (root / class_name).mkdir(parents=True, exist_ok=True)

    # Save canonical classes file
    (root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    # Save mapping scaffold
    mapping = {
        "language_code": args.language_code.upper(),
        "language_name": args.language_name,
        "classes": classes,
        "notes": "Edit this mapping if your raw labels differ from canonical labels"
    }
    with (root / "class_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    # Save capture notes scaffold
    (root / "capture_notes.md").write_text(
        "# Capture Notes\n\n- Camera/device:\n- Location/background:\n- Variations added:\n- Known issues:\n",
        encoding="utf-8",
    )

    write_dataset_readme(
        root=root,
        language_code=args.language_code,
        language_name=args.language_name,
        classes=classes,
        samples_per_class=args.samples_per_class,
    )

    print("=" * 60)
    print("Custom dataset scaffold created")
    print("=" * 60)
    print(f"Root: {root.resolve()}")
    print(f"Language: {args.language_name} ({args.language_code.upper()})")
    print(f"Classes: {len(classes)}")
    print("Files created:")
    print("  - classes.txt")
    print("  - class_mapping.json")
    print("  - capture_notes.md")
    print("  - README.md")
    print("Class folders created for all classes.")


if __name__ == "__main__":
    main()
