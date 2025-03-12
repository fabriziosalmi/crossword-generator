import argparse
import json
import random
import time
from typing import Dict, List, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel


class CrosswordGenerator:
    def __init__(self, width: int, height: int, word_list_file: str, max_attempts: int = 10000, seed: int = None):
        """
        Initializes the CrosswordGenerator.

        Args:
            width: The width of the crossword grid.
            height: The height of the crossword grid.
            word_list_file: Path to the JSON file containing the word list.
            max_attempts: The maximum number of attempts to try generating the crossword.
            seed: Optional seed for the random number generator for reproducible results.
        """
        self.width = width
        self.height = height
        self.grid = [['.' for _ in range(width)] for _ in range(height)]
        self.console = Console()  # Initialize console *before* calling load_word_list
        self.word_list: Dict[int, Dict[str, str]] = self.load_word_list(word_list_file)
        self.placed_words: List[Dict] = []
        self.attempts = 0
        self.max_attempts = max_attempts
        if seed is not None:
            random.seed(seed)


    def load_word_list(self, filename: str) -> Dict[int, Dict[str, str]]:
        """Loads the word list from a JSON file, handling various error cases."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if not isinstance(data, dict) or "words" not in data or not isinstance(data["words"], list):
                    raise ValueError("Invalid JSON structure.  Must be a dict with a 'words' key containing a list.")

                structured_list: Dict[int, Dict[str, str]] = {}
                for word_data in data["words"]:
                    if not isinstance(word_data, dict) or not all(k in word_data for k in ("word", "clue")):
                        raise ValueError("Each word entry must be a dictionary with 'word' and 'clue' keys.")

                    word = word_data["word"].upper()
                    clue = word_data["clue"]
                    if not word.isalpha():
                        raise ValueError(f"Invalid word: '{word}'. Words must contain only letters.")
                    length = len(word)
                    if length not in structured_list:
                        structured_list[length] = {}
                    if word in structured_list[length]:
                        self.console.print(f"[yellow]Warning: Duplicate word '{word}' found.  Using the first entry.[/]")
                    else:
                        structured_list[length][word] = clue
                return structured_list
        except FileNotFoundError:
            self.console.print(f"[red]Error: Word list file '{filename}' not found.[/]")
            return {}  # Return empty dict to allow program termination
        except json.JSONDecodeError:
            self.console.print(f"[red]Error: Invalid JSON format in '{filename}'.[/]")
            return {}
        except ValueError as e:
            self.console.print(f"[red]Error: {e}[/]")
            return {}
        except Exception as e:  # Catch other potential errors
            self.console.print(f"[red]An unexpected error occurred: {e}[/]")
            return {}

    def find_candidate_words(self, row: int, col: int, direction: str, length: int) -> List[Tuple[str, str]]:
        """Finds words from the word list that fit the given pattern."""
        candidates: List[Tuple[str, str]] = []
        if length not in self.word_list:
            return candidates

        pattern = ""
        if direction == "HORIZONTAL":
            if col + length > self.width:  # Check bounds immediately
                return []
            pattern = "".join(self.grid[row][col:col+length])
        else:  # direction == "VERTICAL"
            if row + length > self.height:
                return []
            pattern = "".join(self.grid[row+i][col] for i in range(length))

        for word, clue in self.word_list[length].items():
            if all(pattern[i] == '.' or pattern[i] == word[i] for i in range(length)):
                candidates.append((word, clue))
        return candidates

    def check_constraints(self, word: str, row: int, col: int, direction: str) -> bool:
        """Checks if placing the word violates crossword constraints."""
        length = len(word)
        if direction == "HORIZONTAL":
            if col + length > self.width: return False
            for i in range(length):
                # Check for intersecting words
                if self.grid[row][col + i] != '.' and self.grid[row][col + i] != word[i]:
                    return False
                # Check for adjacent words (no touching rule)
                if row > 0 and self.grid[row - 1][col + i] != '.': return False
                if row < self.height - 1 and self.grid[row + 1][col + i] != '.': return False
            # Check ends for adjacency
            if col > 0 and self.grid[row][col - 1] != '.': return False
            if col + length < self.width and self.grid[row][col + length] != '.': return False

        else:  # direction == "VERTICAL"
            if row + length > self.height: return False
            for i in range(length):
                # Check for intersecting words
                if self.grid[row + i][col] != '.' and self.grid[row + i][col] != word[i]:
                    return False
                # Check for adjacent words
                if col > 0 and self.grid[row + i][col - 1] != '.': return False
                if col < self.width - 1 and self.grid[row + i][col + 1] != '.': return False
            # Check ends for adjacency
            if row > 0 and self.grid[row - 1][col] != '.': return False
            if row + length < self.height and self.grid[row + length][col] != '.': return False

        return True

    def place_word(self, word: str, row: int, col: int, direction: str):
        """Places the word on the grid."""
        for i, letter in enumerate(word):
            if direction == "HORIZONTAL":
                self.grid[row][col + i] = letter
            else:
                self.grid[row + i][col] = letter
        self.placed_words.append({"word": word, "row": row, "col": col, "direction": direction})

    def remove_word(self, word_info: Dict):
        """Removes the word from the grid."""
        for i, letter in enumerate(word_info["word"]):
            if word_info["direction"] == "HORIZONTAL":
                if self.grid[word_info["row"]][word_info["col"] + i] == letter: # only remove if it's the placed letter
                    self.grid[word_info["row"]][word_info["col"] + i] = '.'
            else:
                if self.grid[word_info["row"] + i][word_info["col"]] == letter:
                    self.grid[word_info["row"] + i][word_info["col"]] = '.'
        self.placed_words.remove(word_info)

    def solve(self, row: int = 0, col: int = 0, direction: str = "HORIZONTAL") -> bool:
        """Recursively attempts to solve the crossword."""
        self.attempts += 1
        if self.attempts > self.max_attempts:  # Stop after max_attempts
            return False

        if row == self.height:  # Base case: Grid is full
            return True

        next_row, next_col = (row, col + 1) if col + 1 < self.width else (row + 1, 0)

        if self.grid[row][col] != '.':  # Cell already filled
            return self.solve(next_row, next_col, direction)

        lengths = sorted(self.word_list.keys(), reverse=True) # try longest words first
        for length in lengths:
            for direction in ["HORIZONTAL", "VERTICAL"]:
                candidates = self.find_candidate_words(row, col, direction, length)
                random.shuffle(candidates)  # Randomize candidate order
                for word, _ in candidates:
                    if self.check_constraints(word, row, col, direction):
                        self.place_word(word, row, col, direction)
                        if self.solve(next_row, next_col, direction):
                            return True  # Solution found
                        self.remove_word(self.placed_words[-1])  # Backtrack
        return False

    def get_solution(self) -> List[Dict]:
        """Returns the solution in a structured format."""
        solution = []
        for word_info in self.placed_words:
            word = word_info["word"]
            clue = self.word_list[len(word)][word]  # Retrieve clue
            solution.append({
                "word": word,
                "clue": clue,
                "row": word_info["row"],
                "col": word_info["col"],
                "direction": word_info["direction"],
            })
        return solution

    def print_grid_panel(self, progress) -> Panel:
        """Creates a Panel containing the grid and progress bar."""
        table = Table(show_header=False, show_edge=True)
        for _ in range(self.width):
            table.add_column(width=3, justify="center")

        for row in self.grid:
            colored_row = [
                "[white on black] [/]" if cell == '.' else f"[white on blue]{cell}[/]"
                for cell in row
            ]
            table.add_row(*colored_row)

        panel_content = Table.grid(padding=1)
        panel_content.add_row(table)
        panel_content.add_row(progress)
        return Panel(panel_content, title="Crossword Progress")


    def run_solver(self):
        """Runs the solver with a Rich progress bar and live update."""
        if not self.word_list: # if word list failed to load, exit.
            return

        with Live(console=self.console, refresh_per_second=20) as live:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("Attempts: {task.fields[attempts]}"),
                transient=True,
                console=self.console
            )

            task = progress.add_task("Generating Crossword", attempts=0)
            live.update(self.print_grid_panel(progress))

            start_time = time.time()
            solved = self.solve()
            end_time = time.time()

            if solved:
                progress.update(task, description="[green]Crossword Generated![/]", attempts=self.attempts)
                live.update(self.print_grid_panel(progress))
                self.console.print(f"[green]Crossword generated in {end_time - start_time:.2f} seconds.[/]")
                solution = self.get_solution()
                self.console.print("\n[bold underline]Solution:[/]")
                for entry in solution:
                    self.console.print(
                        f"[cyan]{entry['word']}[/] ([yellow]{entry['direction']}[/]): {entry['clue']} - Start: ([magenta]{entry['row']}[/], [magenta]{entry['col']}[/])"
                    )
            else:
                progress.update(task, description="[red]No solution found.[/]", attempts=self.attempts)
                live.update(self.print_grid_panel(progress))
                self.console.print(f"[red]No solution found after {self.attempts} attempts and {end_time - start_time:.2f} seconds.[/]")


def main():
    parser = argparse.ArgumentParser(description="Generate a crossword puzzle.")
    parser.add_argument("word_list_file", help="Path to the JSON word list file.")
    parser.add_argument("-w", "--width", type=int, default=10, help="Width of the crossword grid.")
    parser.add_argument("-H", "--height", type=int, default=10, help="Height of the crossword grid.")
    parser.add_argument("-m", "--max_attempts", type=int, default=10000, help="Maximum number of attempts.")
    parser.add_argument("-s", "--seed", type=int, help="Seed for the random number generator.")
    args = parser.parse_args()

    generator = CrosswordGenerator(args.width, args.height, args.word_list_file, args.max_attempts, args.seed)
    generator.run_solver()

if __name__ == "__main__":
    main()
