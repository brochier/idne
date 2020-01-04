import numpy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
# https://github.com/zhaocq-nlp/Attention-Visualization/blob/master/exec/plot_heatmap.py



def plot_heat_map(mma, target_labels, source_labels):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False)
    ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False)

    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(source_labels, minor=False)
    # target words -> row labels
    ax.set_yticklabels(target_labels, minor=False)

    plt.xticks(rotation=45)

    # plt.tight_layout()
    plt.show()
