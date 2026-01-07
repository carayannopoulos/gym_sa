import numpy as np
from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
import os
import time


class TSPEnv:
    def __init__(
        self,
        env_params: dict,
        rank: int = 0,
    ):
        self.dim = env_params["dim"]
        self.lims = env_params["lims"]
        self.n_cities = env_params["n_cities"]
        self.state = np.zeros(self.dim)
        self.old_state = np.zeros(self.dim)
        self.best_state = np.zeros(self.dim)
        self.best_objective = -np.inf
        self.rng = np.random.default_rng()
        self.init_cities = env_params["init_cities"]
        self.n_steps = 0
        self.rank = rank
        self.agent_dir = f"agent_{self.rank}"
        os.makedirs(self.agent_dir, exist_ok=True)

        if "render_best_state" not in env_params:
            env_params["render_best_state"] = False

        if "render_delay" in env_params:
            self.render_delay = env_params["render_delay"]
        else:
            self.render_delay = 20000

        self.render_best_state = env_params["render_best_state"]

        # select the locations of the cities at random
        if self.init_cities == "random":
            self.cities = self.rng.uniform(
                self.lims[0], self.lims[1], (self.n_cities, self.dim)
            )
        elif self.init_cities == "coordinates":
            self.cities = env_params["cities"]

        elif self.init_cities == "matrix":
            self.distance_matrix = env_params["distance_matrix"]
            if "cities" in env_params:
                self.cities = env_params["cities"]

            else:
                self.cities = None
                self.render_best_state = False

        if self.n_cities <= 10:
            self.global_opt_state, self.global_opt_objective = self.global_best_state()
            print(f"Global opt state: {self.global_opt_state}")
            print(f"Global opt objective: {self.global_opt_objective}")

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # select a random starting state
        self.state = self.rng.permutation(self.n_cities)
        self.old_state = deepcopy(self.state)

        # Objective: maximize -(state-3)^2 (peak at state=3)
        self.current_objective = self.compute_distance(self.state)

        self.old_objective = self.current_objective

        return self.state, self.current_objective

    def compute_distance(self, state):
        # compute the distance between the cities in the order of the state
        # distance = 0
        # for i in range(self.n_cities - 1):
        #     if self.init_cities == "matrix":
        #         distance += self.distance_matrix[state[i], state[i + 1]]
        #     elif self.init_cities == "coordinates":
        #         distance += np.linalg.norm(
        #             self.cities[state[i]] - self.cities[state[i + 1]]
        #         )
        # if self.init_cities == "matrix":
        #     distance += self.distance_matrix[state[-1], state[0]]
        # elif self.init_cities == "coordinates":
        #     distance += np.linalg.norm(self.cities[state[-1]] - self.cities[state[0]])

        if self.init_cities == "matrix":
            # Vectorized distance calculation using distance matrix
            # Create pairs of consecutive cities (including return to start)
            from_indices = state
            to_indices = np.roll(
                state, -1
            )  # Shift by -1 to get next city, last city connects to first

            # Extract all distances at once
            distance = np.sum(self.distance_matrix[from_indices, to_indices])

        elif self.init_cities == "coordinates":
            # Vectorized distance calculation using coordinates
            # Get the ordered cities according to the state
            ordered_cities = self.cities[state]

            # Calculate distances between consecutive cities
            # Shift by -1 to get next city, last city connects to first
            next_cities = np.roll(ordered_cities, -1, axis=0)

            # Calculate all pairwise distances at once
            distances = np.linalg.norm(ordered_cities - next_cities, axis=1)
            distance = np.sum(distances)

        return -distance

    def two_opt(self, state, edges):

        # perform a 2-opt move on the state
        # select two edges at random
        i = edges[0]
        j = edges[1]

        # make sure the edges are not the same or adjacent
        assert i != j
        assert i != (j - 1) % self.n_cities
        assert i != (j + 1) % self.n_cities

        # perform the 2-opt move
        a = max(i, j)
        b = min(i, j)

        # reverse the order of the cities between the edges
        state[b + 1 : a + 1] = state[b + 1 : a + 1][::-1]

        return state

    def get_perturbation(self, temperature):
        # Propose a small random step
        i = self.rng.integers(0, self.n_cities)
        j = self.rng.integers(0, self.n_cities)

        # make sure the edges are not the same or adjacent
        while j in ((i - 1) % self.n_cities, i, (i + 1) % self.n_cities):
            j = self.rng.integers(0, self.n_cities)

        return (i, j)

    def step(self, perturbation):

        self.n_steps += 1

        # print(f"Perturbation: {perturbation}")

        # Apply perturbation
        self.state = self.two_opt(self.state, perturbation)
        # Objective: minimize the distance
        self.current_objective = self.compute_distance(self.state)

        if self.n_cities <= 10:
            self.current_objective -= self.global_opt_objective

        if self.current_objective > self.best_objective:
            self.best_objective = self.current_objective
            self.best_state = deepcopy(self.state)

            if self.render_best_state and self.n_steps >= self.render_delay:
                self.render()

        done = False
        info = {}
        return deepcopy(self.state), self.current_objective, done, info

    # def restore_old_state(self):
    #     self.state = deepcopy(self.old_state)

    # def update_old_state(self):
    #     self.old_state = deepcopy(self.state)
    #     self.old_objective = self.current_objective

    def set_state(self, state, objective):
        self.state = deepcopy(state)
        self.current_objective = objective

    def global_best_state(self):
        # enumerate all possible states

        opt_state = None
        opt_objective = -np.inf

        if self.n_cities <= 10:
            for perm in itertools.permutations(range(self.n_cities)):
                objective = self.compute_distance(perm)
                if objective > opt_objective:
                    opt_objective = objective
                    opt_state = perm

        else:
            print("Not implemented for n_cities > 10")

        return opt_state, opt_objective

    def generate_distance_matrix(self):
        if self.init_cities == "coordinates" or self.init_cities == "random":
            self.distance_matrix = np.zeros((self.n_cities, self.n_cities))
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    self.distance_matrix[i, j] = np.linalg.norm(
                        self.cities[i] - self.cities[j]
                    )

    def render(self, tag=None):

        if self.cities is not None:
            # plot the cities
            plt.clf()
            plt.scatter(self.cities[:, 0], self.cities[:, 1])
            # add city indices as labels
            for i in range(len(self.cities)):
                plt.annotate(str(i), (self.cities[i, 0], self.cities[i, 1]))
            # plot the state
            plt.plot(self.cities[self.state, 0], self.cities[self.state, 1])
            plt.plot(
                (self.cities[self.state[-1], 0], self.cities[self.state[0], 0]),
                (self.cities[self.state[-1], 1], self.cities[self.state[0], 1]),
            )
            if tag is not None:
                plt.savefig(f"tsp_{tag}.png", dpi=300)
            else:
                plt.savefig("tsp.png", dpi=300)

        else:
            print("Not implemented for init_cities == matrix")


if __name__ == "__main__":

    cities = np.array(
        [
            [1150.0, 1760.0],
            [630.0, 1660.0],
            [40.0, 2090.0],
            [750.0, 1100.0],
            [750.0, 2030.0],
            [1030.0, 2070.0],
            [1650.0, 650.0],
            [1490.0, 1630.0],
            [790.0, 2260.0],
            [710.0, 1310.0],
            [840.0, 550.0],
            [1170.0, 2300.0],
            [970.0, 1340.0],
            [510.0, 700.0],
            [750.0, 900.0],
            [1280.0, 1200.0],
            [230.0, 590.0],
            [460.0, 860.0],
            [1040.0, 950.0],
            [590.0, 1390.0],
            [830.0, 1770.0],
            [490.0, 500.0],
            [1840.0, 1240.0],
            [1260.0, 1500.0],
            [1280.0, 790.0],
            [490.0, 2130.0],
            [1460.0, 1420.0],
            [1260.0, 1910.0],
            [360.0, 1980.0],
        ]
    )

    dm = np.array(
        [
            [
                0,
                107,
                241,
                190,
                124,
                80,
                316,
                76,
                152,
                157,
                283,
                133,
                113,
                297,
                228,
                129,
                348,
                276,
                188,
                150,
                65,
                341,
                184,
                67,
                221,
                169,
                108,
                45,
                167,
            ],
            [
                107,
                0,
                148,
                137,
                88,
                127,
                336,
                183,
                134,
                95,
                254,
                180,
                101,
                234,
                175,
                176,
                265,
                199,
                182,
                67,
                42,
                278,
                271,
                146,
                251,
                105,
                191,
                139,
                79,
            ],
            [
                241,
                148,
                0,
                374,
                171,
                259,
                509,
                317,
                217,
                232,
                491,
                312,
                280,
                391,
                412,
                349,
                422,
                356,
                355,
                204,
                182,
                435,
                417,
                292,
                424,
                116,
                337,
                273,
                77,
            ],
            [
                190,
                137,
                374,
                0,
                202,
                234,
                222,
                192,
                248,
                42,
                117,
                287,
                79,
                107,
                38,
                121,
                152,
                86,
                68,
                70,
                137,
                151,
                239,
                135,
                137,
                242,
                165,
                228,
                205,
            ],
            [
                124,
                88,
                171,
                202,
                0,
                61,
                392,
                202,
                46,
                160,
                319,
                112,
                163,
                322,
                240,
                232,
                314,
                287,
                238,
                155,
                65,
                366,
                300,
                175,
                307,
                57,
                220,
                121,
                97,
            ],
            [
                80,
                127,
                259,
                234,
                61,
                0,
                386,
                141,
                72,
                167,
                351,
                55,
                157,
                331,
                272,
                226,
                362,
                296,
                232,
                164,
                85,
                375,
                249,
                147,
                301,
                118,
                188,
                60,
                185,
            ],
            [
                316,
                336,
                509,
                222,
                392,
                386,
                0,
                233,
                438,
                254,
                202,
                439,
                235,
                254,
                210,
                187,
                313,
                266,
                154,
                282,
                321,
                298,
                168,
                249,
                95,
                437,
                190,
                314,
                435,
            ],
            [
                76,
                183,
                317,
                192,
                202,
                141,
                233,
                0,
                213,
                188,
                272,
                193,
                131,
                302,
                233,
                98,
                344,
                289,
                177,
                216,
                141,
                346,
                108,
                57,
                190,
                245,
                43,
                81,
                243,
            ],
            [
                152,
                134,
                217,
                248,
                46,
                72,
                438,
                213,
                0,
                206,
                365,
                89,
                209,
                368,
                286,
                278,
                360,
                333,
                284,
                201,
                111,
                412,
                321,
                221,
                353,
                72,
                266,
                132,
                111,
            ],
            [
                157,
                95,
                232,
                42,
                160,
                167,
                254,
                188,
                206,
                0,
                159,
                220,
                57,
                149,
                80,
                132,
                193,
                127,
                100,
                28,
                95,
                193,
                241,
                131,
                169,
                200,
                161,
                189,
                163,
            ],
            [
                283,
                254,
                491,
                117,
                319,
                351,
                202,
                272,
                365,
                159,
                0,
                404,
                176,
                106,
                79,
                161,
                165,
                141,
                95,
                187,
                254,
                103,
                279,
                215,
                117,
                359,
                216,
                308,
                322,
            ],
            [
                133,
                180,
                312,
                287,
                112,
                55,
                439,
                193,
                89,
                220,
                404,
                0,
                210,
                384,
                325,
                279,
                415,
                349,
                285,
                217,
                138,
                428,
                310,
                200,
                354,
                169,
                241,
                112,
                238,
            ],
            [
                113,
                101,
                280,
                79,
                163,
                157,
                235,
                131,
                209,
                57,
                176,
                210,
                0,
                186,
                117,
                75,
                231,
                165,
                81,
                85,
                92,
                230,
                184,
                74,
                150,
                208,
                104,
                158,
                206,
            ],
            [
                297,
                234,
                391,
                107,
                322,
                331,
                254,
                302,
                368,
                149,
                106,
                384,
                186,
                0,
                69,
                191,
                59,
                35,
                125,
                167,
                255,
                44,
                309,
                245,
                169,
                327,
                246,
                335,
                288,
            ],
            [
                228,
                175,
                412,
                38,
                240,
                272,
                210,
                233,
                286,
                80,
                79,
                325,
                117,
                69,
                0,
                122,
                122,
                56,
                56,
                108,
                175,
                113,
                240,
                176,
                125,
                280,
                177,
                266,
                243,
            ],
            [
                129,
                176,
                349,
                121,
                232,
                226,
                187,
                98,
                278,
                132,
                161,
                279,
                75,
                191,
                122,
                0,
                244,
                178,
                66,
                160,
                161,
                235,
                118,
                62,
                92,
                277,
                55,
                155,
                275,
            ],
            [
                348,
                265,
                422,
                152,
                314,
                362,
                313,
                344,
                360,
                193,
                165,
                415,
                231,
                59,
                122,
                244,
                0,
                66,
                178,
                198,
                286,
                77,
                362,
                287,
                228,
                358,
                299,
                380,
                319,
            ],
            [
                276,
                199,
                356,
                86,
                287,
                296,
                266,
                289,
                333,
                127,
                141,
                349,
                165,
                35,
                56,
                178,
                66,
                0,
                112,
                132,
                220,
                79,
                296,
                232,
                181,
                292,
                233,
                314,
                253,
            ],
            [
                188,
                182,
                355,
                68,
                238,
                232,
                154,
                177,
                284,
                100,
                95,
                285,
                81,
                125,
                56,
                66,
                178,
                112,
                0,
                128,
                167,
                169,
                179,
                120,
                69,
                283,
                121,
                213,
                281,
            ],
            [
                150,
                67,
                204,
                70,
                155,
                164,
                282,
                216,
                201,
                28,
                187,
                217,
                85,
                167,
                108,
                160,
                198,
                132,
                128,
                0,
                88,
                211,
                269,
                159,
                197,
                172,
                189,
                182,
                135,
            ],
            [
                65,
                42,
                182,
                137,
                65,
                85,
                321,
                141,
                111,
                95,
                254,
                138,
                92,
                255,
                175,
                161,
                286,
                220,
                167,
                88,
                0,
                299,
                229,
                104,
                236,
                110,
                149,
                97,
                108,
            ],
            [
                341,
                278,
                435,
                151,
                366,
                375,
                298,
                346,
                412,
                193,
                103,
                428,
                230,
                44,
                113,
                235,
                77,
                79,
                169,
                211,
                299,
                0,
                353,
                289,
                213,
                371,
                290,
                379,
                332,
            ],
            [
                184,
                271,
                417,
                239,
                300,
                249,
                168,
                108,
                321,
                241,
                279,
                310,
                184,
                309,
                240,
                118,
                362,
                296,
                179,
                269,
                229,
                353,
                0,
                121,
                162,
                345,
                80,
                189,
                342,
            ],
            [
                67,
                146,
                292,
                135,
                175,
                147,
                249,
                57,
                221,
                131,
                215,
                200,
                74,
                245,
                176,
                62,
                287,
                232,
                120,
                159,
                104,
                289,
                121,
                0,
                154,
                220,
                41,
                93,
                218,
            ],
            [
                221,
                251,
                424,
                137,
                307,
                301,
                95,
                190,
                353,
                169,
                117,
                354,
                150,
                169,
                125,
                92,
                228,
                181,
                69,
                197,
                236,
                213,
                162,
                154,
                0,
                352,
                147,
                247,
                350,
            ],
            [
                169,
                105,
                116,
                242,
                57,
                118,
                437,
                245,
                72,
                200,
                359,
                169,
                208,
                327,
                280,
                277,
                358,
                292,
                283,
                172,
                110,
                371,
                345,
                220,
                352,
                0,
                265,
                178,
                39,
            ],
            [
                108,
                191,
                337,
                165,
                220,
                188,
                190,
                43,
                266,
                161,
                216,
                241,
                104,
                246,
                177,
                55,
                299,
                233,
                121,
                189,
                149,
                290,
                80,
                41,
                147,
                265,
                0,
                124,
                263,
            ],
            [
                45,
                139,
                273,
                228,
                121,
                60,
                314,
                81,
                132,
                189,
                308,
                112,
                158,
                335,
                266,
                155,
                380,
                314,
                213,
                182,
                97,
                379,
                189,
                93,
                247,
                178,
                124,
                0,
                199,
            ],
            [
                167,
                79,
                77,
                205,
                97,
                185,
                435,
                243,
                111,
                163,
                322,
                238,
                206,
                288,
                243,
                275,
                319,
                253,
                281,
                135,
                108,
                332,
                342,
                218,
                350,
                39,
                263,
                199,
                0,
            ],
        ]
    )

    dim = cities.shape[1]

    env_params = {
        "dim": dim,
        "lims": (0, 1),
        "n_cities": cities.shape[0],
        "init_cities": "matrix",
        "cities": cities,
        "render_best_state": True,
        "distance_matrix": dm,
    }

    env = TSPEnv(env_params)
    env.reset()

    # print(env.state)
    # print(env.compute_distance(env.state))

    best_tour = (
        np.array(
            [
                1,
                28,
                6,
                12,
                9,
                5,
                26,
                29,
                3,
                2,
                20,
                10,
                4,
                15,
                18,
                17,
                14,
                22,
                11,
                19,
                25,
                7,
                23,
                27,
                8,
                24,
                16,
                13,
                21,
            ]
        )
        - 1
    )
    print(env.compute_distance(best_tour))

    env.state = best_tour
    env.render(tag="bays29")

    env.generate_distance_matrix()
    np.savetxt("distance_matrix.csv", env.distance_matrix, delimiter=",")
