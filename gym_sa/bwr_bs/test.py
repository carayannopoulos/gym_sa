import numpy as np
import os

from bwr_bs.envs import bwr_bs_sa_v1
from bwr_bs.envs.utils.simulate_tools import is_valid_pattern, label_to_index


# env_data = {
#     "core_sheet": "flexible_v2.xlsx",
#     "input_root": "template",
#     "n_cycles": 3,
#     "fue_lab_file": "LP.txt",
#     "template": "sim_files/eq_template_rods_v1.inp",
#     "library": "/home/loukas/phd/core_design/bwr/bwrx300/libraries/bwrx300_v4.lib",
#     "restart": "/home/loukas/phd/core_design/bwr/environments/bwr_bs/bwr_bs/envs/sim_files/c1.res",
#     "SDM_template": "sim_files/sdm_template.inp",
#     "core_power": 972.7e6,
#     "pins_per_bundle": 92,
#     "n_bundles": 240,
#     # constraints
#     "MFLPD": (0.91, 25),
#     "MFLCPR": (1.73, 25),
#     "MAPRAT": (0.91, 25),
#     "max_keff": (1.005, 50),
#     "min_keff": (0.995, 50),
#     "exposure": (62, 25),
#     "enrichment": (4.0, 1),
#     "cycle_length": (700, 0),
#     "enrichment_threshold": 4.5,
#     "enrichment_penalty": -1e1,
#     "layout_penalty": -1.2e1,
#     "lcoe_weight": 10,
#     "SDM_limit": 1.53,
#     "SDM_weight": 0.5,
#     "max_crd_change": 48,
#     "max_avg_crd_change": (10, 5),
#     # curriculum
#     "curriculum": [
#         [
#             "MFLPD",
#             "MAPRAT",
#             "MFLCPR",
#             "exposure",
#             "SDM",
#             "min_keff",
#             "max_keff",
#             "max_avg_crd_change",
#         ],
#         ["lcoe"],
#     ],
#     "thresholds": [0, 100],
#     "assembly_width": 15.24,  # Cold width of physical assembly, cm (DXA)
#     "core_height": 381.0,  # Cold height of active fuel, not including axial reflectors, cm
#     "depletion_points": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13.356],
#     "n_depletion_steps": 27,  # the number of depletion steps will depend on how the template is written
#     "n_rods": 4,
#     "crd_template": "rod_template.txt",
#     "template_sheet": "flexible_v2.xlsx",
#     "episode_length": 100,
#     "symmetry": "quarter",
#     "rod_movement_size": 10,
#     "initial_state": "random",
#     "reset_state": "random",
#     "seeding": False,
#     "crd_file": "crd_input.txt",
#     "fue_lab_file": "LP.txt",
#     "CPR_limit": 1.4,
# }

# os.system("rm -rf bwr_archive")

env_data_sheet = "bs_input_debug.xlsx"

env = bwr_bs_sa_v1(env_data_sheet, exepath="/home/loukas/Simulate3/bin/simulate3")
env.reset()
print(env.eq_representation)

# env.eq_representation = np.array(
#     [
#         ["FA01", "2510", "2912", "2708"],
#         ["FA03", "2316", "2318", "2304"],
#         ["FA04", "2114", "3316", "    "],
#         ["FA01", "2314", "2710", "3112"],
#         ["FA04", "2918", "1914", "2908"],
#         ["FA04", "2914", "2916", "3312"],
#         ["FA04", "1904", "1910", "3518"],
#         ["FA04", "2518", "3114", "3110"],
#         ["FA02", "2118", "2104", "3314"],
#         ["FA04", "3116", "2718", "3514"],
#         ["FA01", "1912", "1906", "2910"],
#         ["FA03", "1916", "2112", "    "],
#         ["FA03", "2110", "1918", "1902"],
#         ["FA02", "3318", "3118", "2102"],
#         ["FA04", "2312", "2116", "2302"],
#         ["FA04", "2514", "2108", "2706"],
#         ["FA03", "2308", "2306", "2506"],
#         ["FA04", "2508", "    ", "    "],
#         ["FA03", "2716", "2504", "3516"],
#         ["FA03", "    ", "    ", "    "],
#         ["FA04", "2714", "    ", "    "],
#         ["FA04", "1908", "    ", "    "],
#         ["FA01", "2106", "2310", "    "],
#         ["FA04", "2712", "    ", "    "],
#         ["FA03", "2512", "    ", "    "],
#         ["FA01", "2516", "    ", "    "],
#         ["FA04", "    ", "    ", "    "],
#         ["FA04", "    ", "    ", "    "],
#         ["FA01", "    ", "    ", "    "],
#     ]
# ).astype(str)

env.eq_representation = np.array(
    [
        ["FA01", "2112", "2912", "2708"],
        ["FA03", "2316", "2318", "2304"],
        ["FA04", "2110", "3316", "3516"],
        ["FA01", "2314", "2710", "3112"],
        ["FA03", "2918", "2916", "2908"],
        ["FA01", "2914", "2718", "3312"],
        ["FA03", "1904", "1906", "3518"],
        ["FA03", "2106", "3114", "3110"],
        ["FA02", "2118", "2104", "3314"],
        ["FA03", "3116", "1914", "3514"],
        ["FA01", "1912", "1918", "2910"],
        ["FA01", "1916", "2510", "2504"],
        ["FA03", "2114", "1910", "1902"],
        ["FA02", "2514", "3118", "2102"],
        ["FA04", "2312", "2308", "2302"],
        ["FA04", "3318", "2108", "2706"],
        ["FA02", "2116", "2306", "2506"],
        ["FA04", "2508", "    ", "    "],
        ["FA03", "2716", "    ", "    "],
        ["FA03", "2310", "    ", "    "],
        ["FA04", "2714", "    ", "    "],
        ["FA01", "1908", "    ", "    "],
        ["FA01", "2518", "    ", "    "],
        ["FA04", "2712", "    ", "    "],
        ["FA03", "2512", "    ", "    "],
        ["FA02", "2516", "    ", "    "],
        ["FA04", "    ", "    ", "    "],
        ["FA03", "    ", "    ", "    "],
        ["FA03", "    ", "    ", "    "],
    ]
).astype(str)

print(env.eq_representation)

mask = env.build_mask()

print(is_valid_pattern(env.eq_representation, env.template))
# print(mask)
a1 = int(input(f"type: "))
a2 = int(input(f"action 1: "))
print(f"mask action 1: {mask['action_1'][a1][a2]}")
m = a2 // 3
n = a2 % 3 + 1
id = env.eq_representation[m][n]
a, b = label_to_index(id, dim=18)
a3 = int(input(f"action 2: "))
m = a3 // 3
n = a3 % 3 + 1
id2 = env.eq_representation[m][n]
# x, y = label_to_index(id2, dim=18)
print(f"mask action 2: {mask['action_2'][a1][a2][a3]}")
print(f"id: {id}; {env.template.loc[a,b]} -- id2: {id2}; {n}")
# a4 = int(input(f"rod step: "))


# print("new loc 1: ", env.eq_representation[a2 // 3, a2 % 3 + 1])
# print("new loc 2: ", env.eq_representation[a3 // 3, a3 % 3 + 1])
# print(env.eq_representation)
# print(f"original: ({a2}, {0}) -- {env.eq_representation[a2, 0]}")
# print(f"new: {env.fresh_bundles.index[a3]}")

# print(f"loc 1: {mask[a1][0][a2]}")
# print(f"loc 2: {mask[a1][1][a2, a3]}")

# print(f"old rod position: {env.control_rods[a2, a3]}")
# print(f"rod movement: {2 * a4 - env.env_data['rod_movement_size']}")

# env.step([a1, a2, a3, a4])
# print(f"new rod position: {env.control_rods[a2, a3]}")


# print(env.core)
