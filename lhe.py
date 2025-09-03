from typing import Optional
import logging
import subprocess
import numpy as np
import os
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
import random 
from datetime import datetime
from utils.utils import *
from utils.llm_client.base import BaseClient
from problems.mvmoe_pomo.eval_train import getproblem


class ReEvo:
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
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.vrp_problem = getproblem()
        
        self.init_prompt()
        self.init_population()


    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type
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
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st.txt') if self.problem_type != "black_box" else file_to_string(f'{self.prompt_dir}/common/user_reflector_st_black_box.txt') # shrot-term reflection
        self.user_reflector_lt_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_lt.txt') # long-term reflection
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        self.minimutation_prompt = file_to_string(f'{self.prompt_dir}/common/minimutation.txt')
        self.mutataion_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
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
        self.print_minimutation_prompt = True


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

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.update_iter()
        
        # Generate responses
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

        #responses = self.generator_llm.multi_chat_completion([messages], self.cfg.init_pop_size, temperature = self.generator_llm.temperature + 0.3) # Increase the temperature for diverse initial population
        #print("responses:", responses)
        responses = []
        for _ in range(self.cfg.init_pop_size):
            response = self.generator_llm.multi_chat_completion([messages], temperature=self.generator_llm.temperature + 0.3)
            responses.append(response[0])

            
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]
        

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
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
            # Store objective value for each individual
            if traceback_msg == '': # If execution has no error
                try:

                    individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                    individual["exec_success"] = True
                except Exception as e:

                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population

    def evaluate_population2(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
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
            # Store objective value for each individual
            if traceback_msg == '': # If execution has no error
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                    individual["exec_success"] = True
                except:
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
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
            logging.info(f"Iteration {self.iteration}, response_id {response_id}: 0bj")
        objs_with_indices = [(i, individual["obj"]) for i, individual in enumerate(population)]
        indices, objs = zip(*objs_with_indices)
        objs = [individual["obj"] for individual in population]
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


    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 3:
            return None
        trial = 0
        while len(selected_population) < 3 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=3, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            parents_list = parents.tolist()
            parents_list=sorted(parents_list,key=lambda x: x["obj"], reverse=True)
            parents = np.array(parents_list, dtype=object)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population


    def random_select2(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 3:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if ind1["obj"] == ind2["obj"]:
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["obj"] > ind2["obj"]:
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])
        
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name = self.func_name,
            func_desc = self.func_desc,
            problem_desc = self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
            )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
                logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code


    def short_term_reflection(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before mini_mutate two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 3):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Asynchronously generate responses
        response_lst = self.reflector_llm.multi_chat_completion(messages_lst)
        return response_lst, worse_code_lst, better_code_lst



    def short_term_reflection2(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term2 reflection before crossover2 two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Asynchronously generate responses
        response_lst = self.reflector_llm.multi_chat_completion(messages_lst)
        return response_lst, worse_code_lst, better_code_lst
    


    def crossover(self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        messages_lst = []
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                func_signature1 = func_signature1,
                worse_code = worse_code,
                better_code = better_code,
                reflection = reflection,
                func_name = self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False
        
        # Asynchronously generate responses
        response_lst = self.generator_llm.multi_chat_completion(messages_lst)
        crossed_population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(response_lst)]

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population
    

        
        

    def mini_mutate(self, population: list[dict],short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        """
        mini_mutate before crossovering1 two individuals.
        """
        messages_lst = []
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        i = 0
        for reflection in zip(reflection_content_lst):
            # Select two individuals
            variant_a = population[i+2]
            varianta_code = filter_code(variant_a["code"])
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            user = self.minimutation_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                varianta_code = varianta_code,
                reflection = reflection,
                func_name = self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)            
            # Print crossover prompt for the first iteration
            if self.print_minimutation_prompt:
                logging.info("Minimutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_minimutation_prompt = False
            i = i + 3

            # Asynchronously generate responses
        response_lst = self.generator_llm.multi_chat_completion(messages_lst)
        minimued_population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(response_lst)]

        assert len(minimued_population) == self.cfg.pop_size
        return minimued_population

                                                                


    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            # Select
            population_to_select = self.population if (self.elitist is None or self.elitist in self.population) else [self.elitist] + self.population 
            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            # Short-term reflection
            short_term_reflection_tuple = self.short_term_reflection(selected_population) # (response_lst, worse_code_lst, better_code_lst)。
            mini_mutate_population=self.mini_mutate(selected_population,short_term_reflection_tuple)
            # Crossover
            crossed_population = self.crossover(short_term_reflection_tuple)
            minimutate_population_eval=self.evaluate_population2(mini_mutate_population)
            crossed_population_eval=self.evaluate_population(crossed_population)
            selected_population2=self.random_select2(minimutate_population_eval+crossed_population_eval)
            #加入Short-term reflection2
            short_term_reflection_tuple2 = self.short_term_reflection2(selected_population2)
            #加入Crossover2
            crossed_population2 = self.crossover(short_term_reflection_tuple2)
            # Evaluate
            self.population.extend(self.evaluate_population(crossed_population2))
            # Update
            self.update_iter()
        return self.best_code_overall, self.best_code_path_overall ,self.vrp_problem
