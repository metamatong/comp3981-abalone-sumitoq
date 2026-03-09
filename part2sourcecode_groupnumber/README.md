# Part 2 Source Code Submission

This folder contains the minimal code and file structure needed to submit the Abalone state-space generator portion of the project.

## Included Files

Only the files needed to parse `Test#.input`, generate legal moves, apply board rules, and write `.board` / `.move` outputs are included:

```text
part2sourcecode_groupnumber/
├── README.md
├── run.py
└── abalone/
    ├── __init__.py
    ├── board.py
    ├── file_handler.py
    ├── state_space.py
    ├── game/
    │   ├── __init__.py
    │   └── board.py
    ├── state_space_inputs/
    │   └── Test#.input files go here
    └── state_space_outputs/
        └── Generated .board / .move files are written here
```

Not included on purpose:

- Web UI files
- Terminal game UI
- AI/minimax player code
- Tests
- Static assets

Those are not required for running the Part 2 state-space generator.

## Build Requirements

- Python 3.8 or newer
- No external packages
- No install step

## How To Run

1. Open a terminal in this folder:

```bash
cd part2sourcecode_groupnumber
```

2. Run the generator on an input file:

```bash
python3 run.py --state-input-file abalone/state_space_inputs/Test1.input
```

This writes:

- `abalone/state_space_outputs/Test1.board`
- `abalone/state_space_outputs/Test1.move`

You can also choose an explicit output path:

```bash
python3 run.py \
  --state-input-file abalone/state_space_inputs/Test1.input \
  --state-output-file abalone/state_space_outputs/Test1.board
```

## Running With Additional `Test#.input` Files

1. Place the new input file inside:

```text
abalone/state_space_inputs/
```

Example:

```text
abalone/state_space_inputs/Test9.input
```

2. Run:

```bash
python3 run.py --state-input-file abalone/state_space_inputs/Test9.input
```

3. The program will generate:

```text
abalone/state_space_outputs/Test9.board
abalone/state_space_outputs/Test9.move
```

## Input File Format

Each input file must contain:

1. First line: `b` or `w`
2. Remaining lines: comma-separated marble tokens such as `C5b,D5b,E7w`

Example:

```text
b
C3b,D3b,E3b,F3w,G3w
```

## Optional Verification

If you already have an expected `.board` file with the same stem inside `abalone/state_space_outputs/`, run:

```bash
python3 run.py --state-input-file abalone/state_space_inputs/Test1.input --state-verify
```

This prints a comparison summary and writes a generated sibling file for inspection.

## Manual Comparison Examples

If you want to compare generated outputs against hand-authored answer files:

```bash
diff -u abalone/state_space_outputs/Test6.answer.board abalone/state_space_outputs/Test6.board
diff -u abalone/state_space_outputs/Test6.answer.move abalone/state_space_outputs/Test6.move
```
