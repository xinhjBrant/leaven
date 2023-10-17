# from src.lean_server import LeanEnv

# server = LeanEnv(do_logging=True)

# content = '''import analysis.special_functions.log.basic

# theorem aime_1983_p1
#   (x y z w : ℕ)
#   (ht : 1 < x ∧ 1 < y ∧ 1 < z)
#   (hw : 0 ≤ w)
#   (h0 : real.log w / real.log x = 24)
#   (h1 : real.log w / real.log y = 40)
#   (h2 : real.log w / real.log (x * y * z) = 12):
#   real.log w / real.log z = 60 
# := sorry '''

# verify_result = server.verify_lean_file(content)
# print(verify_result)
# # Shall return {'error': '', 'warning': "line 3, column 0: declaration 'aime_1983_p1' uses sorry", 'info': '', 'open_states': 'line: 11, column: 3, proof state: x y z w : ℕ,\nht : 1 < x ∧ 1 < y ∧ 1 < z,\nhw : 0 ≤ w,\nh0 : real.log ↑w / real.log ↑x = 24,\nh1 : real.log ↑w / real.log ↑y = 40,\nh2 : real.log ↑w / real.log (↑x * ↑y * ↑z) = 12\n⊢ real.log ↑w / real.log ↑z = 60', 'context': 'import analysis.special_functions.log.basic\n\ntheorem aime_1983_p1\n  (x y z w : ℕ)\n  (ht : 1 < x ∧ 1 < y ∧ 1 < z)\n  (hw : 0 ≤ w)\n  (h0 : real.log w / real.log x = 24)\n  (h1 : real.log w / real.log y = 40)\n  (h2 : real.log w / real.log (x * y * z) = 12):\n  real.log w / real.log z = 60 \n:= sorry '}

# server.close()

def add_mathlib_data():
    from src.proof_search_agent import ProvingSearchAgent
    from tqdm import tqdm
    agent = ProvingSearchAgent()
    agent.load_proving_envs()
    from pathlib import Path
    import json
    for p in tqdm(list(Path('/data2/xinhuajian/MathVoyager/dataset_temp').glob('**/*.json'))):
        with open(p) as f:
            data = json.load(f)
        for k, v in data.items():
            agent.add_theorem_env(name=k, init_context=v['init_context'], pre_lines=v['prelines'], post_lines=v['postlines'])
    agent.close_proving_envs()

def add_minif2f_data():
    from src.proof_search_agent import ProvingSearchAgent
    agent = ProvingSearchAgent()
    agent.load_proving_envs()
    from pathlib import Path
    from tqdm import tqdm
    import json
    with open('/data2/xinhuajian/MathVoyager/minif2f_import.lean') as f:
        pre_lines = f.read()
    with open('/data2/xinhuajian/MathVoyager/minif2f.json') as f:
        data = json.load(f)
    for k, v in tqdm(list(data.items())):
        agent.add_theorem_env(name=k, init_context=v["formal_statement"], pre_lines=pre_lines)
    agent.close_proving_envs()

def test_search():
    from src.proof_search_agent import ProvingSearchAgent
    agent = ProvingSearchAgent()
    agent.load_proving_envs()
    # provide the name of the declaration to initialize the proving environment, e.g. absolute_value.map_units_int
    init_result = agent.init_search('absolute_value.map_units_int')
    # return verify result: '{"error": "", "warning": "line 30, column 0: declaration \'absolute_value.map_units_int\' uses sorry", "info": "", "open_states": ["line: 32, column: 17, proof state: S : Type u_2,\\n_inst_2 : linear_ordered_comm_ring S,\\nabv : absolute_value ℤ S,\\nx : ℤˣ\\n⊢ ⇑abv ↑x = 1"], "context": "/-\\nCopyright (c) 2021 Anne Baanen. All rights reserved.\\nReleased under Apache 2.0 license as described in the file LICENSE.\\nAuthors: Anne Baanen\\n-/\\nimport algebra.module.basic\\nimport algebra.order.absolute_value\\nimport data.int.cast.lemmas\\nimport data.int.units\\nimport group_theory.group_action.units\\n\\n/-!\\n# Absolute values and the integers\\n\\n> THIS FILE IS SYNCHRONIZED WITH MATHLIB4.\\n> Any changes to this file require a corresponding PR to mathlib4.\\n\\nThis file contains some results on absolute values applied to integers.\\n\\n## Main results\\n\\n * `absolute_value.map_units_int`: an absolute value sends all units of `ℤ` to `1`\\n * `int.nat_abs_hom`: `int.nat_abs` bundled as a `monoid_with_zero_hom`\\n-/\\n\\nvariables {R S : Type*} [ring R] [linear_ordered_comm_ring S]\\n\\n\\n@[simp]\\nlemma absolute_value.map_units_int (abv : absolute_value ℤ S) (x : ℤˣ) :\\n  abv x = 1 :=\\n begin  repeat { sorry }  end \\n\\n"}'
    agent.close_proving_envs()
    # provide next proof step, e.g. rcases int.units_eq_one_or x with (rfl | rfl); simp
    run_result = agent.run_tac(0, 'rcases int.units_eq_one_or x with (rfl | rfl); simp')
    # return verify result: '{"error": "", "warning": "", "info": "", "open_states": [], "context": "/-\\nCopyright (c) 2021 Anne Baanen. All rights reserved.\\nReleased under Apache 2.0 license as described in the file LICENSE.\\nAuthors: Anne Baanen\\n-/\\nimport algebra.module.basic\\nimport algebra.order.absolute_value\\nimport data.int.cast.lemmas\\nimport data.int.units\\nimport group_theory.group_action.units\\n\\n/-!\\n# Absolute values and the integers\\n\\n> THIS FILE IS SYNCHRONIZED WITH MATHLIB4.\\n> Any changes to this file require a corresponding PR to mathlib4.\\n\\nThis file contains some results on absolute values applied to integers.\\n\\n## Main results\\n\\n * `absolute_value.map_units_int`: an absolute value sends all units of `ℤ` to `1`\\n * `int.nat_abs_hom`: `int.nat_abs` bundled as a `monoid_with_zero_hom`\\n-/\\n\\nvariables {R S : Type*} [ring R] [linear_ordered_comm_ring S]\\n\\n\\n@[simp]\\nlemma absolute_value.map_units_int (abv : absolute_value ℤ S) (x : ℤˣ) :\\n  abv x = 1 :=\\n begin  rcases int.units_eq_one_or x with (rfl | rfl); simp,  repeat { sorry }   end \\n\\n"}'

    # if not run_result['open_states'] then the proof is finished.

if __name__ == "__main__":
    # add_minif2f_data()
    # add_mathlib_data()
    # test_search()
    from src.lean_manager import checking_complete
    checking_complete()