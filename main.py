import hydra
import logging 
import os
from pathlib import Path
import subprocess
from utils.utils import init_client
import time


ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)
ifvrp = False

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    client = init_client(cfg)

    if not ifvrp :
        #if cfg.algorithm == "reevo":
            #from eoh import EoH as LHH
            #from reevo import ReEvo as LHH
            #from reevo1 import ReEvo as LHH
            #from randomgeneration import Randombaseline as LHH
        #elif cfg.algorithm == "ael":
            #from baselines.ael.ga import AEL as LHH
        #elif cfg.algorithm == "eoh":
            #from baselines.eoh import EoH as LHH
        #elif cfg.algorithm == "random":
            #from baselines.randomgeneration import Randombaseline as LHH
        #else:
            #raise NotImplementedError
        # Main algorithm
        #lhh = LHH(cfg, ROOT_DIR, client)
        #best_code_overall, best_code_path_overall = lhh.evolve()
        #logging.info(f"Best Code Overall: {best_code_overall}")
        #logging.info(f"Best Code Path Overall: {best_code_path_overall}")
        # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
        #with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w') as file:
            #file.writelines(best_code_overall + '\n')
        test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
        test_script_stdout = "best_code_overall_val_stdout.txt"
        logging.info(f"Running validation script...: {test_script}")
        with open(test_script_stdout, 'w') as stdout:
            subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
        logging.info(f"Validation script finished. Results are saved in {test_script_stdout}.")
    
        # Print the results
        with open(test_script_stdout, 'r') as file:
            for line in file.readlines():
                logging.info(line.strip())
    else :

        if cfg.algorithm == "reevo":
            #from reevo2 import ReEvo as LHH
            #from eoh2 import EoH as LHH
            from reevo_mvmoe import ReEvo as LHH
            #from randomgeneration2 import Randombaseline as LHH
        elif cfg.algorithm == "ael":
            from baselines.ael.ga import AEL as LHH
        elif cfg.algorithm == "eoh":
            from baselines.eoh import EoH as LHH
        else:
            raise NotImplementedError
        # Main algorithm
        start = time.time()
        lhh = LHH(cfg, ROOT_DIR, client)
        best_code_overall, best_code_path_overall , vrp_problem = lhh.evolve()
        logging.info(f"Best Code Overall: {best_code_overall}")
        logging.info(f"Best Code Path Overall: {best_code_path_overall}")
        total_time = time.time() - start
        #Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
        #with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/{vrp_problem}/gpt.py", 'w') as file:
            #file.writelines(best_code_overall + '\n')
        #test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval_test.py"
        #test_script_stdout = "best_code_overall_val_stdout.txt"
        #logging.info(f"Running validation script...: {test_script}")
        
        #with open(test_script_stdout, 'w') as stdout:
            #subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
        
        logging.info(f"Time: {total_time:.3f} s")
        #logging.info(f"Validation script finished. Results are saved in {test_script_stdout}.")
        # Print the results
        #with open(test_script_stdout, 'r') as file:
            #for line in file.readlines():
                #logging.info(line.strip())
        #logging.info(f"Time: {total_time:.3f} s")
if __name__ == "__main__":
    main()