<a href="https://github.com/alexandrainst/tiny_language_detection">
<img
 src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/alexandra/alexandra-logo.jpeg"
 width="239"
 height="175"
 align="right"
 alt="Alexandra Institute Logo"
/>
</a>

# Tiny Language Detection

Audio language detection for edge devices, distinguishing between Danish and English speech. This project is part of the REINS research initiative, exploring how to scale language detection models to resource-constrained hardware.

See [PLAN.md](PLAN.md) for the experimental roadmap and detailed implementation plan.

______________________________________________________________________
[![License](https://img.shields.io/github/license/alexandrainst/tiny_language_detection)](https://github.com/alexandrainst/tiny_language_detection/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/tiny_language_detection)](https://github.com/alexandrainst/tiny_language_detection/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/tiny_language_detection/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Dan Saattrup Smart (<dan.smart@alexandra.dk>)

## Setup

### Installation

1. Run `make install`, which sets up a virtual environment and all Python dependencies
   therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Adding and Removing Packages

To install new PyPI packages, run:

```bash
uv add <package-name>
```

To remove them again, run:

```bash
uv remove <package-name>
```

To show all installed packages, run:

```bash
uv pip list
```

## All Built-in Commands

The project includes the following convenience commands:

- `make install`: Install the project and its dependencies in a virtual environment.
- `make check`: Lint and format the code using `ruff`, and type check using `ty`.
- `make tree`: Show the project structure as a tree.

## A Word on Modules and Scripts

In the `src` directory there are two subdirectories, `tiny_language_detection`
and `scripts`. This is a brief explanation of the differences between the two.

### Modules

All Python files in the `tiny_language_detection` directory are _modules_
internal to the project package. Examples here could be a general data loading script,
a definition of a model, or a training function. Think of modules as all the building
blocks of a project.

When a module is importing functions/classes from other modules we use the _relative
import_ notation - here's an example:

```python
from .other_module import some_function
```

### Scripts

Python files in the `scripts` folder are scripts, which are short code snippets that
are _external_ to the project package, and which is meant to actually run the code. As
such, _only_ scripts will be called from the terminal. An analogy here is that the
internal `numpy` code are all modules, but the Python code you write where you import
some `numpy` functions and actually run them, that a script.

When importing module functions/classes when you're in a script, you do it like you
would normally import from any other package:

```python
from tiny_language_detection import some_function
```
