from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import time
import numpy as np
import string
import random
import copy
import math
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import mean_squared_error


def get_target_image(filename):
    real_image = Image.open(filename)
    def convert_to_black_and_white(x): return 1 if x > 128 else 0
    b_w_image = real_image.convert('L').point(
        convert_to_black_and_white, mode='1')
    b_w_image = np.array(b_w_image, dtype=int)
    return b_w_image


def resolution_to_string_length(height, width):
    return int(width / 6) * int(height / 10)


def get_font_and_related_dimensions(font_file_path, example_string, image_width):
    font = ImageFont.truetype(font_file_path, size=10)
    line_height = font.getsize(example_string)[1]
    symbols_per_row = int(
        image_width / (font.getsize(example_string)[0] / len(example_string)))

    return font, line_height, symbols_per_row


PRINTABLE_CHARS = string.ascii_letters + \
    string.punctuation + '                       '
#PRINTABLE_CHARS = ' @'


def generate_random_string(length):
    return ''.join(random.choices(PRINTABLE_CHARS, k=length))


def create_random_population(size, string_length):
    population = np.array([generate_random_string(string_length)
                          for _ in range(size)])
    return population


class EvalData():
    def __init__(self, target_image, font, line_height, symbols_per_row, worst_case) -> None:
        self.target_image = target_image
        self.font = font
        self.line_height = line_height
        self.symbols_per_row = symbols_per_row
        self.worst_case = worst_case
        self.to_eval = None


# def text_to_image(text, width, height, font, line_height, symbols_per_row, ImageFont=ImageFont, ImageDraw=ImageDraw, Image=Image, np=np, int=int):
#     lines = [text[i:i+symbols_per_row]
#              for i in range(0, len(text), symbols_per_row)]

#     img = Image.new("1", (width, height), "#FFF")
#     draw = ImageDraw.Draw(img)

#     y = 0
#     for line in lines:
#         draw.text((0, y), line, "#000", font=font)
#         y += line_height

#     return np.array(img, dtype=int)

def text_to_image(text, width, height, font, line_height, symbols_per_row, ImageFont=ImageFont, ImageDraw=ImageDraw, Image=Image, np=np, int=int):
    lines = [text[i:i+symbols_per_row]
             for i in range(0, len(text), symbols_per_row)]

    vertical_padding = (height - len(lines) * line_height) / 2
    horizontal_padding = (width - len(lines[0]) * 6) / 2

    img = Image.new("1", (width, height), "#FFF")
    draw = ImageDraw.Draw(img)

    y = vertical_padding
    for line in lines:
        draw.text((horizontal_padding, y), line, "#000", font=font)
        y += line_height

    return np.array(img, dtype=int)


def evaluate_ascii_string(eval_data):
    size = eval_data.target_image.shape
    ascii_image = text_to_image(
        eval_data.to_eval, size[1], size[0], eval_data.font, eval_data.line_height, eval_data.symbols_per_row)

    return mean_squared_error(ascii_image, eval_data.target_image)


def evaluate_ascii_strings(children, bare_eval_data):
    evaluation_data = []
    for child in children:
        data = copy.copy(bare_eval_data)
        data.to_eval = child
        evaluation_data.append(data)

    fitnesses = Parallel(n_jobs=cpu_count())(
        delayed(evaluate_ascii_string)(e) for e in evaluation_data)
    return fitnesses


def step(selection_function, crossover_function, mutation_function, population, fitnesses, fitness_history, eval_data, iteration_percentage):
    """
    Creates a new generation from the previous generation, using the operators
    provided during object creation. Also tracks the best fitness in every
    generation. Returns children and fitnesses.
    """
    selection = selection_function(population, fitnesses)
    children = selection if crossover_function is None else crossover_function(
        selection)
    children = [mutation_function(e, iteration_percentage) for e in children]
    children_fitnesses = evaluate_ascii_strings(children, eval_data)
    combined_population = np.append(np.array(population), np.array(children))
    combined_fitnesses = np.append(
        np.array(fitnesses), np.array(children_fitnesses))
    amount = len(population)
    indices_of_survivors = np.argpartition(combined_fitnesses, amount)[:amount]
    children = [e.tolist() for e in combined_population[indices_of_survivors]]
    fitnesses = [e.tolist() for e in combined_fitnesses[indices_of_survivors]]
    fitness_history.append(np.min(fitnesses))
    population = copy.copy(children)

    return population, fitnesses, fitness_history


def select(population, fitnesses, percent=20):
    s_population = [s[0] for s in sorted(
        [p for p in zip(population, fitnesses)], key=lambda x: x[1])]
    new_population = s_population[:int(
        np.floor((len(population) * percent)))-1] * int((1 / percent))
    new_population.extend(s_population[:len(population) - len(new_population)])

    return new_population


def cut_point_crossover(population):

    # we have to shuffle the population as nearby orderings are the clones of each other
    np.random.shuffle(population)

    individual_length = len(population[0])
    # for every two parents, there should be two children
    new_population = []

    # implementation of alternating position crossover for a pair of parents.
    def crossover(a, b, individual_length):
        cut_point = np.random.randint(0, individual_length)
        children = [a[:cut_point] + b[cut_point:],
                    b[:cut_point] + a[cut_point:]]
        return children

    # perform crossover twice per pair, alternating which parent to start with.
    for i in range(len(population) // 2):
        parent_a, parent_b = population[i], population[i+1]

        # only apply crossover for a percentage of the population
        # if np.random.random() < 0.5:
        new_population += crossover(parent_a, parent_b, individual_length)
        # else:
        #new_population += [parent_a, parent_b]

    return new_population


# def mutate(individual):

#     size = len(individual)

#     mutated_part_size = np.random.randint(
#         size / 20, size / 5) if np.random.random() < 0.1 else int(size / 100)
#     mutated_part_start_index = np.random.randint(0, size - mutated_part_size)
#     mutated_individual = individual[:mutated_part_start_index] + generate_random_string(
#         mutated_part_size) + individual[mutated_part_start_index + mutated_part_size:]

#     return mutated_individual

def mutate(individual, iteration_percentage):

    size = len(individual)

    mutated_part_size = np.random.randint(
        size / 20, size / 5) if np.random.random() < 0.1 else int(size / 100 * (1.01 - iteration_percentage))
    mutated_part_start_index = np.random.randint(0, size - mutated_part_size)
    mutated_individual = individual[:mutated_part_start_index] + generate_random_string(
        mutated_part_size) + individual[mutated_part_start_index + mutated_part_size:]

    return mutated_individual

# def mutate(individual, iteration_percentage=0):

#     size = len(individual)
#     mutated_individual = []
#     for character in individual:
#         mutated_individual.append(
#             character if np.random.random() < 0.2 else generate_random_string(1))

#     return ''.join(mutated_individual)


# there's a pil bug to convert images from bool array, this makes it grayscale first. Only needed to display, not compare.
def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


if __name__ == "__main__":
    filename = 'smile.png'
    target_image = get_target_image(filename)
    ascii_length = resolution_to_string_length(*(target_image.shape))
    font, line_height, symbols_per_row = get_font_and_related_dimensions(
        'Courier Prime.ttf', generate_random_string(ascii_length), target_image.shape[1])

    # Parameters for the experiment
    # MAX_NO_OF_ITERATIONS = 2
    # PRINT_EVERY_N_ITERATIONS = 100
    # POPULATION_SIZE = 10

    WORST_CASE_EVALUATION = mean_squared_error(text_to_image(generate_random_string(
        ascii_length), target_image.shape[0], target_image.shape[1], font, line_height, symbols_per_row), target_image)

    eval_data = EvalData(target_image, font, line_height,
                         symbols_per_row, WORST_CASE_EVALUATION)

    population_sizes = np.arange(100, 2100, 100, dtype=int)
    print(population_sizes)
    TOTAL_EVALUATIONS = 1000_000
    fitness_scores = np.empty(len(population_sizes))
    iterations = [int(np.ceil(TOTAL_EVALUATIONS / pop))
                  for pop in population_sizes]

    fitness_history = []

    for i, (pop, it) in enumerate(zip(population_sizes, iterations)):
        start_population = create_random_population(pop, ascii_length)
        start_fitnesses = evaluate_ascii_strings(start_population, eval_data)
        print(pop, it, pop * it)
        population = copy.copy(start_population)
        fitnesses = copy.copy(start_fitnesses)
        for gen_no in range(it):
            population, fitnesses, _ = step(
                select, cut_point_crossover, mutate, population, fitnesses, fitness_history, eval_data, gen_no / it)
        best_solution_index = np.argmin(fitnesses)
        best_fitness = fitnesses[best_solution_index]
        fitness_scores[i] = best_fitness
        print(best_fitness)
        np.save(f'fitness_scores_{i}', fitness_scores)

    print(fitness_scores)
    np.save('fitness_scores', fitness_scores)

    # population = create_random_population(POPULATION_SIZE, ascii_length)
    # fitnesses = evaluate_ascii_strings(population, eval_data)
    # fitness_history = []

    # DIR_NAME = f'./{filename}_{POPULATION_SIZE}_{MAX_NO_OF_ITERATIONS}'
    # if not os.path.exists(DIR_NAME):
    #     os.makedirs(DIR_NAME)
    # else:
    #     raise Exception(
    #         f"Directory {DIR_NAME} exists, rename it to avoid data loss")

    # Running the EA
    # for gen_no in range(MAX_NO_OF_ITERATIONS):
    #     population, fitnesses, fitness_history = step(
    #         select, cut_point_crossover, mutate, population, fitnesses, fitness_history, eval_data, gen_no / MAX_NO_OF_ITERATIONS)

    # if gen_no % PRINT_EVERY_N_ITERATIONS == 0:
    #     print(
    #         f"At generation {gen_no}. Best current fitness: {np.min(fitnesses)}")
    #     best_solution_index = np.argmin(fitnesses)
    #     best_solution = population[best_solution_index]
    #     solution_image = text_to_image(
    #         best_solution, target_image.shape[1], target_image.shape[0], font, line_height, symbols_per_row)
    #     ascii = img_frombytes(solution_image)
    # ascii.save(f"{DIR_NAME}/result_after_{gen_no}_iterations.jpeg")

    # print(fitness_history)
    # # # Printing and plotting results
    # # best_solution_index = np.argmin(fitnesses)
    # # best_solution = population[best_solution_index]
    # # best_fitness = fitnesses[best_solution_index]
    # # print(f"Best solution from EA: {best_fitness} MSE")

    # # ascii = img_frombytes(text_to_image(
    # #     best_solution, target_image.shape[1], target_image.shape[0], font, line_height, symbols_per_row))
    # # ascii.save(f"{DIR_NAME}/result.jpeg")
    # np.save(f"fitness_history", np.array(fitness_history))

    # with open(f"{DIR_NAME}/result.txt", "w") as text_file:
    #     lines = [best_solution[i:i+symbols_per_row]
    #              for i in range(0, len(best_solution), symbols_per_row)]
    #     for line in lines:
    #         text_file.write(f"{line}\n")
