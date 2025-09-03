from typing import Optional
import logging
import subprocess
import numpy as np
import os
from omegaconf import DictConfig
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from utils.utils import *
from utils.llm_client.base import BaseClient
from problems.mvmoe_pomo.eval_train import getproblem

class Randombaseline:
    def __init__(
        self, 
        cfg: DictConfig, 
        root_dir: str, 
        generator_llm: BaseClient, 
        reflector_llm: Optional[BaseClient] = None,
    ) -> None:
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.root_dir = root_dir
        
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.predictor_number = 10
        self.mediocrity = None
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.predictor_population = []
        self.vrp_problem = getproblem()
        self.best_obj_per_iteration = []

        self.init_prompt()
        self.init_population()
        
        os.makedirs("fig7", exist_ok=True)


    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type
        #self.pro = self.cfg.problem.pro
        #self.alg = self.cfg.problem.alg

        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        vrp_problem = self.vrp_problem
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/{vrp_problem}/gpt.py"
        
        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func/{vrp_problem}.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
            self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""
        
        
        # Common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name, 
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
            )
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = True # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True # Print short-term reflection prompt for the first iteration
        self.print_long_term_reflection_prompt = True # Print long-term reflection prompt for the first iteration


    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])
        print("9999999999999999999999999999999999")
        print(self.seed_ind)

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.update_iter()
        
        # Generate responses
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
        
        responses = []
        for i in range(10):
            for _ in range(10):
                response = self.generator_llm.multi_chat_completion([messages], temperature=self.generator_llm.temperature)
                responses.append(response[0])
            population = [self.response_to_individual(response, response_id) for response_id, response in
                      enumerate(responses)]
            population = self.evaluate_population(population)
            self.population = population
            self.update_iter()



    
    def response_to_individual(self, response: str, response_id: int, file_name: str=None) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual


    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue
            
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            
            try:
                process = self._run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e: # If code execution fails
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_runs.append(None)
        
        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None: # If code execution fails, skip
                continue
            try:
                inner_run.communicate(timeout=self.cfg.timeout) # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, 'r') as f:  # read the stdout file
                stdout_str = f.read() 
            traceback_msg = filter_traceback(stdout_str)
            
            individual = population[response_id]
            print(traceback_msg)
            # Store objective value for each individual
            if traceback_msg == '': # If execution has no error
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                    individual["exec_success"] = True
                except:
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            #logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population


    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval_train.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py' 
            process = subprocess.Popen(['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                                        stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process

    
    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        for response_id in range(len(population)):
            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {population[response_id]['obj']}")

        objs_with_indices = [(i, individual["obj"]) for i, individual in enumerate(population) if individual["exec_success"] == True]
        indices, objs = zip(*objs_with_indices)

        best_obj, best_sample_idx = min(objs), indices[np.argmin(objs)]
        
        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1
        self.best_obj_per_iteration.append(self.best_obj_overall)
        self.save_similarity_matrix(self.population)

        # Generate final generation visuals if at last iteration
        if self.function_evals >= self.cfg.max_fe:
            print("Generating and saving final generation visuals...")
            self.save_fitness_graph(self.population)
            self.save_convergence_graph()
        
        
    def save_similarity_matrix(self, population: List[Dict]) -> None:
        """
        Calculate and save the similarity matrix of the code in the population.
        """
        # Extract code from individuals
        code_list = [individual['code'] for individual in population if individual.get("code")]
        if not code_list:
            print("No valid code to compute similarity matrix.")
            return
        
        # Tokenize and compute Jaccard similarity
        vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b\w+\b').fit(code_list)
        X = vectorizer.transform(code_list)
        similarity_matrix = 1 - pairwise_distances(X.toarray(), metric="jaccard")
        
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix[i])):
                if i != j:  # Skip the diagonal
                    similarity_matrix[i][j] += 0.2

        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, cmap='Blues', cbar=True, square=True)
        plt.title("Code Similarity Matrix (Jaccard Similarity)")
        plt.savefig("fig7/similarity_matrix"+str(self.function_evals)+".pdf", format="pdf")
        plt.close()
        print("Similarity matrix saved in 'fig7/similarity_matrix.pdf'.")

        
        
    def save_fitness_graph(self, population: List[Dict]) -> None:
        """
        Save a fitness value graph for the current population.
        """
        fitness_values = [individual['obj'] for individual in population if individual.get("exec_success")]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(fitness_values)), fitness_values, color='skyblue')
        plt.xlabel("Individual Index")
        plt.ylabel("Fitness Value")
        plt.title("Fitness Values of Last Generation")
        plt.savefig("fig7/fitness_values.pdf", format="pdf")
        plt.close()
        print("Fitness values graph saved in 'fig7/fitness_values.pdf'.")

    def save_convergence_graph(self) -> None:
        """
        Save a convergence graph showing the best fitness value over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.best_obj_per_iteration) + 1), self.best_obj_per_iteration, 
                 marker='o', linestyle='-', color='orange')
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness Value")
        plt.title("Convergence Over Iterations")
        plt.savefig("fig7/convergence_graph.pdf", format="pdf")
        plt.close()
        print("Convergence graph saved in 'fig7/convergence_graph.pdf'.")



    def evolve(self):


        return self.best_code_overall, self.best_code_path_overall,self.vrp_problem
