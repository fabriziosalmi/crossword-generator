import argparse
import json
import random
import time
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout


class CrosswordGenerator:
    def __init__(self, width: int, height: int, word_list_file: str, max_attempts: int = 50000, seed: int = None,
                 black_cell_percentage: float = 0.15, clue_reuse_weight: float = 0.3):
        """
        Initializes the CrosswordGenerator with advanced features.

        Args:
            width: Width of the crossword grid.
            height: Height of the crossword grid.
            word_list_file: Path to the JSON file containing the word list.
            max_attempts: Maximum number of attempts for grid generation.
            seed: Optional seed for the random number generator.
            black_cell_percentage: Target percentage of black cells.
            clue_reuse_weight:  Probability (0.0-1.0) of reusing parts of definitions.
        """
        self.width = width
        self.height = height
        self.grid = [['.' for _ in range(width)] for _ in range(height)]
        self.console = Console()
        self.word_list: Dict[int, List[Tuple[str, str]]] = self.load_word_list(word_list_file)  # List of (word, clue) tuples
        self.placed_words: List[Dict] = []
        self.attempts = 0
        self.max_attempts = max_attempts
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.black_cell_percentage = black_cell_percentage
        self.clue_reuse_weight = clue_reuse_weight
        self.layout = Layout() # for displaying both grid and clues
        self.best_grid: List[List[str]] = [['.' for _ in range(width)] for _ in range(height)] # Store the grid with most words
        self.best_placed_words: List[Dict] = []
        self.best_score: float = -1.0 # Number of words placed / total possible

    def load_word_list(self, filename: str) -> Dict[int, List[Tuple[str, str]]]:
        """Loads and structures the word list, handling errors robustly."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if not isinstance(data, dict) or "words" not in data or not isinstance(data["words"], list):
                    raise ValueError("Invalid JSON: Must contain a 'words' key with a list of word entries.")

                structured_list: Dict[int, List[Tuple[str, str]]] = {}
                for word_data in data["words"]:
                    if not isinstance(word_data, dict) or not all(k in word_data for k in ("word", "clue")):
                        raise ValueError("Each word entry must be a dictionary with 'word' and 'clue'.")

                    word = word_data["word"].upper()
                    clue = word_data["clue"]

                    if not word.isalpha():
                        self.console.print(f"[yellow]Skipping invalid word: '{word}' (non-alphabetic characters).[/]")
                        continue  # Skip to the next word

                    length = len(word)
                    if length not in structured_list:
                        structured_list[length] = []
                    if any(w == word for w, _ in structured_list[length]):
                        self.console.print(f"[yellow]Warning: Duplicate word '{word}'. Using first entry.[/]")
                    else:
                        structured_list[length].append((word, clue))
                return structured_list

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            self.console.print(f"[red]Error loading word list: {e}[/]")
            return {} # Return empty, allowing program termination

    def find_candidate_words(self, row: int, col: int, direction: str, length: int) -> List[Tuple[str, str, Optional[str]]]:
        """Finds candidate words and considers reusing parts of clues."""

        candidates: List[Tuple[str, str, Optional[str]]] = []
        if length not in self.word_list:
            return candidates

        pattern = self.get_pattern(row, col, direction, length)
        if not pattern:  # Out of bounds
            return []

        for word, clue in self.word_list[length]:
            if all(pattern[i] == '.' or pattern[i] == word[i] for i in range(length)):
                new_clue = None
                if random.random() < self.clue_reuse_weight:
                   new_clue = self.generate_new_clue(word, clue)
                candidates.append((word, clue, new_clue))  # Store original and potentially new clue
        return candidates

    def generate_new_clue(self, word: str, original_clue: str) -> Optional[str]:
        """Attempts to create a new clue by using parts of the original definition."""
        parts = original_clue.split()
        if len(parts) > 3: # Need enough parts to create new clues
            num_parts_to_use = random.randint(1, len(parts) - 1)  # Use at least 1, but not all
            start_index = random.randint(0, len(parts) - num_parts_to_use)
            new_clue = " ".join(parts[start_index : start_index + num_parts_to_use])
            # Add some variations to make it more clue-like
            if random.random() < 0.3:
                new_clue = f"Part of: {new_clue}"
            elif random.random() < 0.3:
                new_clue = f"Like a {new_clue}"
            elif random.random() < 0.3:
                new_clue = new_clue.capitalize() + "..."

            return new_clue
        return None

    def get_pattern(self, row: int, col: int, direction: str, length: int) -> Optional[str]:
        """Gets the current pattern of letters and blanks for a potential word."""
        if direction == "HORIZONTAL":
            if col + length > self.width:
                return None  # Out of bounds
            return "".join(self.grid[row][col:col + length])
        else:  # direction == "VERTICAL"
            if row + length > self.height:
                return None
            return "".join(self.grid[row + i][col] for i in range(length))

    def check_constraints(self, word: str, row: int, col: int, direction: str) -> bool:
        """Checks for valid placement, including stricter adjacency rules."""
        length = len(word)
        if direction == "HORIZONTAL":
            if col + length > self.width: return False
            # Check for surrounding black cells and parallel words
            for i in range(length):
                if self.grid[row][col + i] != '.' and self.grid[row][col + i] != word[i]:
                    return False  # Overlapping with a different letter
                # Stricter adjacency checks
                if row > 0 and self.grid[row - 1][col + i] not in ('.', word[i]):  # Check above
                    return False
                if row < self.height - 1 and self.grid[row + 1][col + i] not in ('.', word[i]):  # Check below
                    return False

            # Check ends (before and after)
            if (col > 0 and self.grid[row][col - 1] != '.') or \
               (col + length < self.width and self.grid[row][col + length] != '.'):
                return False

        else:  # VERTICAL
            if row + length > self.height: return False

            for i in range(length):
                if self.grid[row + i][col] != '.' and self.grid[row + i][col] != word[i]:
                    return False
                if col > 0 and self.grid[row + i][col - 1] not in ('.', word[i]):
                    return False
                if col < self.width - 1 and self.grid[row + i][col + 1] not in ('.', word[i]):
                    return False

            if (row > 0 and self.grid[row - 1][col] != '.') or \
               (row + length < self.height and self.grid[row + length][col] != '.'):
                return False

        return True

    def place_word(self, word: str, row: int, col: int, direction: str, clue: Optional[str] = None):
        """Places the word on the grid and updates placed_words."""
        for i, letter in enumerate(word):
            if direction == "HORIZONTAL":
                self.grid[row][col + i] = letter
            else:
                self.grid[row + i][col] = letter
        # Use the provided clue, if any, otherwise fetch the original
        used_clue = clue if clue is not None else self.get_original_clue(word)
        self.placed_words.append({"word": word, "row": row, "col": col, "direction": direction, "clue": used_clue})


    def get_original_clue(self, word: str) -> str:
        """Retrieves the original clue for a word from the word list."""
        for _, words in self.word_list.items():
            for w, clue in words:
                if w == word:
                    return clue
        return "Clue not found"  # Should not happen, but handle for safety

    def remove_word(self, word_info: Dict):
        """Removes the word from the grid and placed_words."""
        for i, letter in enumerate(word_info["word"]):
            if word_info["direction"] == "HORIZONTAL":
                # Only remove if it's part of the word being removed, not an intersection
                if self.grid[word_info["row"]][word_info["col"] + i] == letter:
                    self.grid[word_info["row"]][word_info["col"] + i] = '.'
            else:
                if self.grid[word_info["row"] + i][word_info["col"]] == letter:
                    self.grid[word_info["row"] + i][word_info["col"]] = '.'
        self.placed_words.remove(word_info)
        
    def calculate_grid_score(self) -> float:
        """Calculates a score for the current grid based on word placement and black cells."""
        num_placed_words = len(self.placed_words)
        max_possible_words = (self.width * self.height) 
        if max_possible_words == 0: # avoid div by zero
            return 0.0
        
        score = float(num_placed_words) / max_possible_words
        return score

    def solve(self) -> bool:
        """Recursive backtracking solver with improved logic and black cell placement."""
        self.attempts += 1
        if self.attempts > self.max_attempts:
            return False
            
        current_score = self.calculate_grid_score()
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_grid = [row[:] for row in self.grid]  # Deep copy
            self.best_placed_words = self.placed_words[:] # and placed words

        # Find the next empty cell, trying to fill the most constrained first
        next_spot = self.find_most_constrained_spot()
        if next_spot is None:  # Grid is full (either all words or all black cells)
            return True

        row, col, direction = next_spot

        # Try placing words
        lengths = sorted(self.word_list.keys(), reverse=True)  # Longest words first
        for length in lengths:
            for direction in ["HORIZONTAL", "VERTICAL"]:
                candidates = self.find_candidate_words(row, col, direction, length)
                random.shuffle(candidates)  # Shuffle for variety
                for word, original_clue, new_clue in candidates:
                    if self.check_constraints(word, row, col, direction):
                        self.place_word(word, row, col, direction, new_clue)
                        if self.solve():
                            return True  # Solution found
                        self.remove_word(self.placed_words[-1])  # Backtrack

        return False
    

    def find_most_constrained_spot(self) -> Optional[Tuple[int, int, str]]:
        """Finds the best empty cell to fill next, prioritizing constrained spots."""
        best_spot = None
        min_candidates = float('inf')

        # Iterate in a way that prioritizes central and intersecting locations
        for row in list(range(self.height // 2, self.height)) + list(range(self.height // 2 - 1, -1, -1)):
            for col in list(range(self.width // 2, self.width)) + list(range(self.width // 2 - 1, -1, -1)):
                if self.grid[row][col] == '.':
                    for direction in ["HORIZONTAL", "VERTICAL"]:
                        # Count possible words for both directions
                        max_len = self.width - col if direction == "HORIZONTAL" else self.height - row
                        num_candidates = 0
                        for length in range(1,max_len+1): # Check all valid lengths
                            num_candidates += len(self.find_candidate_words(row, col, direction, length))

                        if num_candidates < min_candidates:
                            min_candidates = num_candidates
                            best_spot = (row, col, direction) # Store direction as well

        return best_spot
    
    def initial_black_cell_placement(self):
        """Places initial black cells, aiming for symmetry and connectivity."""
        num_black_cells = int(self.width * self.height * self.black_cell_percentage)

        # Attempt symmetrical placement
        placed = 0
        attempts = 0
        while placed < num_black_cells and attempts < self.max_attempts * 2: # Limit attempts
            attempts += 1
            row = random.randint(0, self.height - 1)
            col = random.randint(0, self.width - 1)

            # Central symmetry:
            sym_row = self.height - 1 - row
            sym_col = self.width - 1 - col
            if self.grid[row][col] == '.' and self.grid[sym_row][sym_col] == '.':
                    self.grid[row][col] = '#' # black cell
                    self.grid[sym_row][sym_col] = '#'
                    placed += 2

        # Ensure connectivity - add black cells near other blacks
        for _ in range(num_black_cells // 2): # Add a few more for better distribution
            row, col = random.choice(self.get_neighbors_of_type('#')) # Find cells around existing black
            if row is not None and col is not None and self.grid[row][col] == '.':
                self.grid[row][col] = '#'

    def get_neighbors_of_type(self, cell_type:str) -> List[Tuple[int,int]]:
        """Find cells around the given type cells"""
        neighbors = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == cell_type:
                    # Check adjacent
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            neighbors.append((nr, nc))
        return neighbors
                            
    def get_solution(self) -> List[Dict]:
        """Returns the solution (placed words and clues) in a structured format."""
        solution = []
        for word_info in self.best_placed_words:  # Use best_placed_words
            solution.append({
                "word": word_info["word"],
                "clue": word_info["clue"],
                "row": word_info["row"],
                "col": word_info["col"],
                "direction": word_info["direction"],
            })
        return solution

    def create_grid_table(self) -> Table:
        """Creates a Rich Table object representing the crossword grid."""
        table = Table(show_header=False, show_edge=True)
        for _ in range(self.width):
            table.add_column(width=3, justify="center")

        for row in self.best_grid:  # Use the best grid
            colored_row = [
                "[black on white]  [/]" if cell == '#' else "[white on black] [/]" if cell == '.' else f"[bold white on black]{cell}[/]"
                for cell in row
            ]
            table.add_row(*colored_row)
        return table

    def create_clues_table(self, solution: List[Dict]) -> Table:
        """Creates a Rich Table for the across and down clues."""
        clues_table = Table(title="Clues", show_header=True, header_style="bold magenta")
        clues_table.add_column("Number", style="cyan", width=6)
        clues_table.add_column("Direction", style="green", width=8)
        clues_table.add_column("Clue", style="yellow")

        # Sort clues for display
        across_clues = sorted([c for c in solution if c["direction"] == "HORIZONTAL"], key=lambda x: (x['row'], x['col']))
        down_clues = sorted([c for c in solution if c["direction"] == "VERTICAL"], key=lambda x: (x['col'], x['row']))


        clue_number = 1
        for clues, direction_label in [(across_clues, "Across"), (down_clues, "Down")]:
            for clue_info in clues:
                clues_table.add_row(str(clue_number), direction_label, clue_info["clue"])
                clue_number += 1

        return clues_table

    def run_solver(self):
        """Runs the solver with a Rich progress bar and live updates."""
        if not self.word_list:
            return  # Exit if word list loading failed

        self.layout.split(
            Layout(name="grid", size=self.height+4),  # Adjust size as needed
            Layout(name="clues")
        )
        self.initial_black_cell_placement() # place black cells first
        with Live(self.layout, console=self.console, refresh_per_second=10) as live:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("Attempts: {task.fields[attempts]} , Best Score: {task.fields[best_score]:.2f}"),
                transient=True,
                console=self.console
            )
            task = progress.add_task("Generating Crossword", attempts=0, best_score = self.best_score)
            self.layout["grid"].update(self.create_grid_table())

            start_time = time.time()
            solved = self.solve()  # Main solving call
            end_time = time.time()

            solution = self.get_solution()
            clues_table = self.create_clues_table(solution)
            self.layout["clues"].update(clues_table)

            if solved or self.best_score > 0:
                if solved:
                    progress.update(task, description="[green]Crossword Generated![/]", attempts=self.attempts, best_score = self.best_score)
                else:
                     progress.update(task, description="[yellow]Partial Crossword Generated (Best Effort).[/]", attempts=self.attempts, best_score=self.best_score)
                self.layout["grid"].update(self.create_grid_table())
                self.console.print(f"[green]Crossword generated in {end_time - start_time:.2f} seconds.[/]")


            else:
                progress.update(task, description="[red]No solution found.[/]", attempts=self.attempts,best_score = self.best_score)
                self.layout["grid"].update(self.create_grid_table()) #show best try
                self.console.print(f"[red]No solution found after {self.attempts} attempts and {end_time - start_time:.2f} seconds.[/]")



def main():
    parser = argparse.ArgumentParser(description="Generate a crossword puzzle.")
    parser.add_argument("word_list_file", help="Path to the JSON word list file.")
    parser.add_argument("-w", "--width", type=int, default=10, help="Width of the grid.")
    parser.add_argument("-H", "--height", type=int, default=10, help="Height of the grid.")
    parser.add_argument("-m", "--max_attempts", type=int, default=50000, help="Maximum number of attempts.")
    parser.add_argument("-s", "--seed", type=int, help="Seed for the random number generator.")
    parser.add_argument("-b", "--black_cells", type=float, default=0.15,
                        help="Approximate percentage of black cells (0.0 - 1.0).")
    parser.add_argument("-c", "--clue_reuse", type=float, default=0.3,
                        help="Probability of reusing parts of definitions as clues (0.0 - 1.0).")

    args = parser.parse_args()

    if not 0.0 <= args.black_cells <= 1.0:
        raise ValueError("Black cell percentage must be between 0.0 and 1.0.")
    if not 0.0 <= args.clue_reuse <= 1.0:
        raise ValueError("Clue reuse probability must be between 0.0 and 1.0.")

    generator = CrosswordGenerator(args.width, args.height, args.word_list_file, args.max_attempts, args.seed,
                                    args.black_cells, args.clue_reuse)
    generator.run_solver()


if __name__ == "__main__":
    main()
