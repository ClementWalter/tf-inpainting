import seaborn as sns
import matplotlib.pyplot as plt


def plot_solution(solution):
    fig, axes = plt.subplots(2, 2)
    sns.heatmap(solution["Vx"].squeeze(), ax=axes[0, 0])
    axes[0, 0].set_title("Vx")
    sns.heatmap(solution["Vy"].squeeze(), ax=axes[0, 1])
    axes[0, 1].set_title("Vy")
    sns.heatmap(solution["Geom"].squeeze(), ax=axes[1, 0])
    axes[1, 0].set_title("Geom")
    sns.heatmap(solution["Pressure"].squeeze(), ax=axes[1, 1])
    axes[1, 1].set_title("Pressure")
    plt.show()
