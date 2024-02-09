import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Simulation:
    def __init__(self, seed, num_firing, num_nonfire, effect=0.5, threshold=0.05, show_plot=False):
        self.seed = seed
        self.num_firing = num_firing
        self.num_nonfire = num_nonfire
        self.effect = effect
        self.threshold = threshold
        self.show_plot = show_plot

    def simulate(self):
        np.random.seed(self.seed)
        
        m0, s0, n0 = 0, 1, 100  # Control group parameters
        m1, s1, n1 = self.effect, 1, 100  # Treatment group parameters

        p_value_fire = [sm.stats.ttest_ind(np.random.normal(m0, s0, size=n0),
                                            np.random.normal(m1, s1, size=n1))[1]
                                            for _ in range(self.num_firing)]

        p_value_nonfire = np.random.uniform(0, 1, size=self.num_nonfire)

        p_values = p_value_fire + p_value_nonfire

        hist_data = [p_value_fire, p_value_nonfire]
        plt.hist(hist_data, bins=30, alpha=0.5, label=['firing', 'non-firing'], stacked=True)
        plt.title('Distribution of uncorrected p-values')
        plt.xlabel('p-value')
        plt.ylabel('Frequency')
        plt.legend()
        if self.show_plot:
            plt.show()

        return p_values

class Evaluation:
    def __init__(self, p_values, significant_indices, threshold=0.05):
        self.p_values = p_values
        self.significant_indices = significant_indices
        self.threshold = threshold

    def evaluate(self):
        significant_p = [self.p_values[index] for index in self.significant_indices]
        significant_p_fire = [p for p in significant_p if p < self.threshold]
        return len(significant_p_fire) / len(self.significant_indices)


# Run simulations and evaluations
sig_fire_results = []
for i in range(5): 
    seed = i
    sim = Simulation(seed, 9500, 500, effect=0.01, threshold=0.05, show_plot=False)
    p_values = sim.simulate()
    significant_indices = [index for index, p in enumerate(p_values) if p < 0.05]
    evaluator = Evaluation(p_values, significant_indices, threshold=0.05)
    sig_fire_results.append(evaluator.evaluate())

power = np.mean(sig_fire_results)
std_dev = np.std(sig_fire_results)

print(f"Power: {power}\nStandard Deviation: {std_dev}")
