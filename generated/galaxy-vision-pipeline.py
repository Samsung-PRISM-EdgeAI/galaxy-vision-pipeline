import torch
import random
import re
import math
import io
import contextlib

"""
FunFuzz: An LLM-Powered Evolutionary Fuzzing Framework

Source Paper: http://arxiv.org/abs/2605.02789v1

Mathematical Idea:
FunFuzz combines evolutionary algorithms with large language models (LLMs)
to generate diverse and challenging inputs for a target system, aiming to
uncover bugs, performance bottlenecks, or correctness issues. The core idea
is an iterative optimization process:

1.  **Initialization:** Start with a population of seed input programs.
2.  **Evaluation:** Each program is executed against the target system
    (e.g., `galaxy-vision-pipeline`). Metrics like latency, correctness,
    or resource usage are collected.
3.  **Fitness Calculation:** A fitness score is assigned to each program,
    quantifying how "interesting" or "problematic" its execution was. Higher
    fitness indicates a more desirable outcome (e.g., high latency,
    correctness failure).
4.  **Selection:** Programs with higher fitness are more likely to be
    selected as "parents" for the next generation.
5.  **Mutation (LLM-Powered):** An LLM is used to intelligently mutate
    selected parent programs. The LLM receives the parent's code and
    feedback (e.g., performance metrics, error messages) and generates
    a new, modified program. This leverages the LLM's understanding of
    code and problem domains to create more effective mutations than
    random byte flips.
6.  **Replacement:** The new, mutated programs form the next generation,
    potentially combined with an "elitism" strategy where the best
    performing programs from the previous generation are carried over.
7.  **Iteration:** Steps 2-6 are repeated for a specified number of
    generations, continuously evolving the population towards inputs
    that expose issues in the target system.

Key Hyperparameters and Default Values:
-   `population_size` (int): Number of programs in each generation. Default: 20
-   `generations` (int): Number of evolutionary steps. Default: 10
-   `elitism_count` (int): Number of best programs to carry over directly to the next generation. Default: 2
-   `mutation_rate` (float): Probability that a selected parent will undergo LLM-powered mutation. Default: 0.8
-   `target_latency_ms` (float): Latency threshold (in milliseconds) above which a program starts gaining fitness points. Default: 100.0
-   `correctness_issue_bonus` (float): Additional fitness points awarded if a correctness issue is found. Default: 1000.0
-   `llm_temperature` (float): Controls the randomness of LLM mutations (higher = more random). Default: 0.7
"""

# --- Mock Target System: galaxy-vision-pipeline ---

def galaxy_vision_pipeline(image_tensor: torch.Tensor) -> dict:
    """
    A mock function representing the NPU-optimized Vision Transformer inference pipeline.

    This function simulates the behavior of a vision pipeline, accepting an
    image tensor and returning performance and correctness metrics. It
    introduces artificial latency and correctness issues based on input
    characteristics to allow the fuzzer to find "problematic" inputs.

    Parameters
    ----------
    image_tensor : torch.Tensor
        The input image tensor, expected shape (batch_size, channels, height, width).

    Returns
    -------
    dict
        A dictionary containing:
        - 'latency': float, simulated inference latency in milliseconds.
        - 'correctness_issue_found': bool, True if a correctness issue was detected.

    Raises
    ------
    ValueError
        If the input tensor has an invalid number of dimensions.
    """
    if image_tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (N, C, H, W), got {image_tensor.ndim}D.")

    batch_size, channels, height, width = image_tensor.shape
    simulated_latency = 50.0  # Base latency in ms
    correctness_issue = False

    # Simulate latency based on image size and complexity
    simulated_latency += (height * width * batch_size) / (224 * 224 * 1) * 20.0  # Larger images take longer
    if image_tensor.min() < -0.5 or image_tensor.max() > 1.5:
        simulated_latency += 30.0 # Out-of-range pixel values might cause overhead

    # Simulate correctness issues
    # 1. Extreme pixel values
    if image_tensor.min() < -0.1 or image_tensor.max() > 1.1:
        correctness_issue = True
        # print(f"DEBUG: Correctness issue: extreme pixel values detected (min={image_tensor.min():.2f}, max={image_tensor.max():.2f})")

    # 2. Unusual dimensions (e.g., very small or very large, non-standard aspect ratio)
    if height < 16 or width < 16 or height > 1024 or width > 1024:
        correctness_issue = True
        # print(f"DEBUG: Correctness issue: unusual dimensions detected (H={height}, W={width})")

    # 3. Non-finite values (NaN, Inf) - though torch.rand won't produce these, LLM might inject
    if not torch.isfinite(image_tensor).all():
        correctness_issue = True
        # print(f"DEBUG: Correctness issue: non-finite values detected")

    # Add some random noise to latency for realism
    simulated_latency += random.uniform(-5, 5)

    return {
        'latency': max(0.0, simulated_latency), # Latency cannot be negative
        'correctness_issue_found': correctness_issue
    }

# --- Mock LLM for Program Mutation ---

class MockLLM:
    """
    A mock Large Language Model designed to mutate Python program strings.

    This mock LLM simulates the behavior of an actual LLM by parsing a
    predefined program template and intelligently modifying its parameters
    based on provided feedback. It aims to generate new programs that
    might expose issues in the target pipeline.
    """

    PROGRAM_TEMPLATE = """
import torch
import random

def generate_input():
    batch_size = {batch_size}
    channels = {channels}
    height = {height}
    width = {width}
    pixel_min = {pixel_min}
    pixel_max = {pixel_max}
    random_seed = {random_seed}
    
    torch.manual_seed(random_seed)
    
    tensor = torch.rand(batch_size, channels, height, width) * (pixel_max - pixel_min) + pixel_min
    return tensor
"""
    # Regex to extract parameters from the program string
    PARAM_REGEX = {
        'batch_size': r"batch_size = (\d+)",
        'channels': r"channels = (\d+)",
        'height': r"height = (\d+)",
        'width': r"width = (\d+)",
        'pixel_min': r"pixel_min = ([-+]?\d*\.?\d+)",
        'pixel_max': r"pixel_max = ([-+]?\d*\.?\d+)",
        'random_seed': r"random_seed = (\d+)",
    }

    def __init__(self, temperature: float = 0.7):
        """
        Initializes the MockLLM.

        Parameters
        ----------
        temperature : float, optional
            Controls the randomness of mutations. Higher values lead to more
            drastic changes. Must be between 0.0 and 1.0. Default is 0.7.
        """
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self.temperature = temperature

    def _extract_params(self, program_code: str) -> dict:
        """
        Extracts parameters from a program string using regex.

        Parameters
        ----------
        program_code : str
            The Python code string to parse.

        Returns
        -------
        dict
            A dictionary of extracted parameter names and their values.
        """
        params = {}
        for name, pattern in self.PARAM_REGEX.items():
            match = re.search(pattern, program_code)
            if match:
                # Convert to appropriate type (int for dimensions, float for pixels)
                if name in ['batch_size', 'channels', 'height', 'width', 'random_seed']:
                    params[name] = int(match.group(1))
                else:
                    params[name] = float(match.group(1))
            else:
                # Fallback to default if not found (shouldn't happen with template)
                if name == 'batch_size': params[name] = 1
                elif name == 'channels': params[name] = 3
                elif name == 'height': params[name] = 224
                elif name == 'width': params[name] = 224
                elif name == 'pixel_min': params[name] = 0.0
                elif name == 'pixel_max': params[name] = 1.0
                elif name == 'random_seed': params[name] = 42
        return params

    def _apply_mutation(self, params: dict, feedback: str) -> dict:
        """
        Applies mutations to parameters based on feedback and temperature.

        Parameters
        ----------
        params : dict
            The dictionary of current program parameters.
        feedback : str
            A string containing feedback from the target pipeline evaluation.

        Returns
        -------
        dict
            A new dictionary with mutated parameters.
        """
        mutated_params = params.copy()
        
        # Determine mutation intensity based on temperature
        mutation_strength = 1 + self.temperature * 2 # 1 to 3x base mutation

        # Prioritize mutations based on feedback
        if "latency" in feedback.lower() and "high" in feedback.lower():
            # Try increasing dimensions or batch size to stress pipeline
            if random.random() < 0.7 * self.temperature: # Higher chance with higher temp
                mutated_params['height'] = int(mutated_params['height'] * (1 + 0.1 * mutation_strength))
                mutated_params['width'] = int(mutated_params['width'] * (1 + 0.1 * mutation_strength))
            if random.random() < 0.5 * self.temperature:
                mutated_params['batch_size'] = int(mutated_params['batch_size'] * (1 + 0.2 * mutation_strength))
            
        if "correctness issue" in feedback.lower():
            # Try extreme pixel values or unusual dimensions
            if random.random() < 0.8 * self.temperature:
                # Introduce out-of-range pixel values
                mutated_params['pixel_min'] = random.uniform(-1.0, -0.1) * mutation_strength
                mutated_params['pixel_max'] = random.uniform(1.1, 2.0) * mutation_strength
            if random.random() < 0.6 * self.temperature:
                # Introduce very small or very large dimensions
                if random.random() < 0.5:
                    mutated_params['height'] = random.choice([4, 8, 16, 512, 1024])
                    mutated_params['width'] = random.choice([4, 8, 16, 512, 1024])
                else: # Introduce highly asymmetric dimensions
                    mutated_params['height'] = random.choice([32, 64, 128])
                    mutated_params['width'] = random.choice([512, 1024])
            if random.random() < 0.3 * self.temperature:
                # Try to make pixel_min > pixel_max (should result in empty range, potentially problematic)
                if mutated_params['pixel_min'] < mutated_params['pixel_max']:
                    mutated_params['pixel_min'], mutated_params['pixel_max'] = \
                        mutated_params['pixel_max'], mutated_params['pixel_min'] + random.uniform(0.1, 0.5)

        # General random mutations if no specific feedback or for diversity
        if random.random() < (0.3 + 0.7 * self.temperature): # Always some random mutation
            param_to_mutate = random.choice(list(self.PARAM_REGEX.keys()))
            
            if param_to_mutate in ['height', 'width', 'batch_size', 'channels']:
                # Mutate dimensions/batch size
                change = random.randint(-5, 5) * int(mutation_strength)
                mutated_params[param_to_mutate] = max(1, mutated_params[param_to_mutate] + change)
            elif param_to_mutate in ['pixel_min', 'pixel_max']:
                # Mutate pixel ranges
                change = random.uniform(-0.2, 0.2) * mutation_strength
                mutated_params[param_to_mutate] += change
                mutated_params[param_to_mutate] = round(mutated_params[param_to_mutate], 2)
            elif param_to_mutate == 'random_seed':
                mutated_params[param_to_mutate] = random.randint(0, 10000)

        # Ensure some basic constraints (e.g., dimensions > 0)
        mutated_params['batch_size'] = max(1, mutated_params['batch_size'])
        mutated_params['channels'] = max(1, min(4, mutated_params['channels'])) # Keep channels reasonable
        mutated_params['height'] = max(1, mutated_params['height'])
        mutated_params['width'] = max(1, mutated_params['width'])
        
        return mutated_params

    def mutate(self, program_code: str, feedback: str = "") -> str:
        """
        Mutates a given program code string based on feedback.

        Parameters
        ----------
        program_code : str
            The Python code string of the program to mutate.
        feedback : str, optional
            A string containing feedback from the target pipeline evaluation,
            used to guide the mutation process. Defaults to an empty string.

        Returns
        -------
        str
            The new, mutated Python code string.
        """
        params = self._extract_params(program_code)
        mutated_params = self._apply_mutation(params, feedback)
        
        # Reconstruct the program string with mutated parameters
        new_program_code = self.PROGRAM_TEMPLATE.format(**mutated_params)
        return new_program_code

# --- FuzzedProgram Class ---

class FuzzedProgram:
    """
    Represents a single program (test case) in the evolutionary fuzzing process.

    Attributes
    ----------
    code : str
        The Python source code of the program.
    fitness : float
        The calculated fitness score of the program, indicating how
        "interesting" or problematic its execution was. Higher is better.
    last_result : dict or None
        The last evaluation result from the target pipeline, or None if not yet evaluated.
    generation : int
        The generation number in which this program was created or last mutated.
    """
    def __init__(self, code: str, generation: int = 0):
        """
        Initializes a FuzzedProgram instance.

        Parameters
        ----------
        code : str
            The Python source code of the program.
        generation : int, optional
            The generation number this program belongs to. Defaults to 0.
        """
        self.code = code
        self.fitness = 0.0
        self.last_result = None
        self.generation = generation

    def __repr__(self):
        """Return a string representation of the FuzzedProgram."""
        return (f"FuzzedProgram(fitness={self.fitness:.2f}, "
                f"gen={self.generation}, code_len={len(self.code)})")

# --- FunFuzz Core Algorithm ---

class FunFuzz:
    """
    Implements the FunFuzz evolutionary fuzzing framework.

    This class orchestrates the evolutionary process, using an LLM to
    generate and mutate input programs for a target function, aiming to
    discover performance or correctness issues.
    """

    def __init__(self,
                 target_pipeline: callable,
                 llm_model: MockLLM,
                 population_size: int = 20,
                 generations: int = 10,
                 elitism_count: int = 2,
                 mutation_rate: float = 0.8,
                 target_latency_ms: float = 100.0,
                 correctness_issue_bonus: float = 1000.0,
                 initial_seed_programs: list = None):
        """
        Initializes the FunFuzz framework.

        Parameters
        ----------
        target_pipeline : callable
            The function to be fuzzed (e.g., `galaxy_vision_pipeline`).
            It should accept a `torch.Tensor` and return a dict with
            'latency' (float) and 'correctness_issue_found' (bool).
        llm_model : MockLLM
            An instance of the LLM used for program mutation.
        population_size : int, optional
            The number of programs in each generation. Defaults to 20.
        generations : int, optional
            The total number of evolutionary generations to run. Defaults to 10.
        elitism_count : int, optional
            The number of top-performing programs to carry over directly
            to the next generation without mutation. Defaults to 2.
        mutation_rate : float, optional
            The probability that a selected parent program will undergo
            LLM-powered mutation. Defaults to 0.8.
        target_latency_ms : float, optional
            The latency threshold (in milliseconds) above which a program
            starts accumulating fitness points for high latency. Defaults to 100.0.
        correctness_issue_bonus : float, optional
            The additional fitness points awarded if a correctness issue
            is detected by the target pipeline. Defaults to 1000.0.
        initial_seed_programs : list of str, optional
            A list of initial Python program strings to seed the first
            generation. If None, a default seed program is used.
        """
        if not callable(target_pipeline):
            raise TypeError("target_pipeline must be a callable function.")
        if not isinstance(llm_model, MockLLM):
            raise TypeError("llm_model must be an instance of MockLLM.")
        if not (0 < population_size):
            raise ValueError("population_size must be positive.")
        if not (0 <= elitism_count < population_size):
            raise ValueError("elitism_count must be between 0 and population_size-1.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0.")

        self.target_pipeline = target_pipeline
        self.llm_model = llm_model
        self.population_size = population_size
        self.generations = generations
        self.elitism_count = elitism_count
        self.mutation_rate = mutation_rate
        self.target_latency_ms = target_latency_ms
        self.correctness_issue_bonus = correctness_issue_bonus
        self.initial_seed_programs = initial_seed_programs if initial_seed_programs is not None else [
            MockLLM.PROGRAM_TEMPLATE.format(
                batch_size=1, channels=3, height=224, width=224,
                pixel_min=0.0, pixel_max=1.0, random_seed=