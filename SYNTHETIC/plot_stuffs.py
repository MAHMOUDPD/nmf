import matplotlib.pyplot as plt
import numpy as np
import os

def perf_plot(histories,methods, prob,rank,output_dir):
    LINE_STYLES = [':', '-.', '-', '--']
    # COLORS = ['k', 'm', 'r', 'b', 'g', 'c','y']
    MARKERS = [
        'o', 's', '^', 'v',
        'D', '*', 'x', '+',
        'p', 'h', '<', '>'
    ]
    LINE_WIDTH = 3.5          # uniform line thickness
    FIG_SIZE = (12, 6)         # same width × height for every figure
    DPI = 300                 # resolution

    # Font sizes
    AXES_FONT_SIZE = 10
    # LABEL_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 14
    # TITLE_FONT_SIZE = 18

    fig, ax = plt.subplots(1, 2, figsize=FIG_SIZE)
    colors = {"QRPBB": "blue", "NMPBB": "red", "ANPBB": "green", \
            "ADPBB-E": "purple", "HALS": "orange","QRPBB-N": "cyan",\
                "NMPBB-N": "magenta", "EPMPBB": "brown", "EPMPBB-W": "pink",\
                    "NeNMF": "black","AdaPGE1": "darkorange","AdaPGE2": "purple", \
                        "AdaPGE3": "darkgreen", "AdAPE4": "olive","NeNMF2": "darkgreen",\
                        "AdAPE3_": "purple", "AdAPE4_": "blue" }

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    for idx, (name, _) in enumerate(methods):
        # 1. Average Residual vs Outer iterations  (now padded, averaged, and with std bands!)
        max_len = max(len(res) for res in histories[name]["res"])
        padded_res = np.array([np.pad(res, (0, max_len - len(res)), constant_values=res[-1]) for res in histories[name]["res"]])
        padded_times = np.array([np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) for t in histories[name]["time"]])
        
        mean_time = np.mean(padded_times, axis=0)
        mean_res = np.mean(padded_res, axis=0)
        std_res = np.std(padded_res, axis=0)
        # loglog semilogy
        ax[0].semilogy(range(max_len), mean_res, label=name, color=colors[name],
                    linestyle=LINE_STYLES[idx % len(LINE_STYLES)], linewidth=LINE_WIDTH, 
                    marker=MARKERS[idx%len(MARKERS)])
        ax[0].fill_between(range(max_len), mean_res - std_res, mean_res + std_res, color=colors[name], alpha=0.1)    
        
        # 2. Average Residual vs Time  (now padded, averaged, and with std bands!)
        max_len = max(len(res) for res in histories[name]["res"])
        padded_res = np.array([np.pad(res, (0, max_len - len(res)), constant_values=res[-1]) for res in histories[name]["res"]])
        padded_times = np.array([np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) for t in histories[name]["time"]])
        
        mean_time = np.mean(padded_times, axis=0)
        mean_res = np.mean(padded_res, axis=0)
        std_res = np.std(padded_res, axis=0)
        
        ax[1].semilogy(mean_time, mean_res, label=name, color=colors[name],
                    linestyle=LINE_STYLES[idx % len(LINE_STYLES)], linewidth=LINE_WIDTH,
                    marker=MARKERS[idx%len(MARKERS)])
        ax[1].fill_between(mean_time, mean_res - std_res, mean_res + std_res, color=colors[name], alpha=0.1)
    # ============================RESIDUALS===========================

    # Title (underscores escaped for proper rendering)
    # title_str = dataset_name.replace('_', r'\_')
    # ax.set_title(f'{title_str}-Convergence', fontsize=TITLE_FONT_SIZE, fontweight='bold')

    fig.tight_layout()

    ax[0].set_xlabel("OuIter")
    ax[0].set_ylabel("Relative Residual")
    ax[0].set_title(f"Residual vs OuIter" )# (Avg over ({NUM_TRIALS}) Seeds)")
    ax[0].set_xlim(left=0)
    ax[0].tick_params(axis='both', labelsize=AXES_FONT_SIZE)
    ax[0].yaxis.set_minor_locator(plt.NullLocator())
    ax[0].legend(
                fontsize=LEGEND_FONT_SIZE,
                loc='best',
                frameon=True,
                facecolor=(0.95, 0.95, 0.95))
    ax[0].grid(True, linestyle='--', alpha=0.3)
    ax[0].set_axisbelow(True)


    ax[1].set_xlabel("Time (Seconds)")
    ax[1].set_ylabel("Relative Residual")
    ax[1].set_title(f"Residual vs Time") # (Avg over ({NUM_TRIALS}) Seeds)")
    ax[1].set_xlim(left=0)
    ax[1].tick_params(axis='both', labelsize=AXES_FONT_SIZE)
    ax[1].yaxis.set_minor_locator(plt.NullLocator())
    ax[1].legend(
                fontsize=LEGEND_FONT_SIZE,
                loc='best',
                frameon=True,
                facecolor=(0.95, 0.95, 0.95))
    ax[1].grid(True, linestyle='--', alpha=0.3)
    ax[1].set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    m,n=prob
    # Save
    # output_dir = 'plots_conlogis_new'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{m}_{n}_{rank}_convergence.png')
    fig.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def perf_plot_convtest(histories,methods, prob,output_dir):
    LINE_STYLES = [':', '-.', '-', '--']
    # COLORS = ['k', 'm', 'r', 'b', 'g', 'c','y']
    MARKERS = [
        'o', 's', '^', 'v',
        'D', '*', 'x', '+',
        'p', 'h', '<', '>'
    ]
    LINE_WIDTH = 2.5          # uniform line thickness
    FIG_SIZE = (6, 5)         # same width × height for every figure
    DPI = 300                 # resolution

    # Font sizes
    AXES_FONT_SIZE = 8
    # LABEL_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 14
    # TITLE_FONT_SIZE = 18

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    colors = {"QRPBB": "blue", "NMPBB": "red", "ANPBB": "green", \
            "ADPBB-E": "purple", "HALS": "orange","QRPBB-N": "cyan",\
                "NMPBB-N": "magenta", "EPMPBB": "brown", "EPMPBB-W": "pink",\
                    "NeNMF": "black","AdaPGE1": "darkorange","AdaPGE2": "purple", \
                        "AdaPGE3": "darkgreen", "AdaPGE4": "olive","NeNMF2": "darkgreen",\
                        "AdAPE3_": "purple", "AdAPE4_": "blue" }

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax[1].spines['top'].set_visible(False)
    # ax[1].spines['right'].set_visible(False)

    for idx, (name, _) in enumerate(methods):
        # 1. Average Residual vs Outer iterations  (now padded, averaged, and with std bands!)
        max_len = max(len(res) for res in histories[name]["res"])
        padded_res = np.array([np.pad(res, (0, max_len - len(res)), constant_values=res[-1]) for res in histories[name]["res"]])
        padded_times = np.array([np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) for t in histories[name]["time"]])
        
        mean_time = np.mean(padded_times, axis=0)
        mean_res = np.mean(padded_res, axis=0)
        std_res = np.std(padded_res, axis=0)
        # loglog
        ax.semilogy(range(max_len), mean_res, label=name, color=colors[name],
                    linestyle=LINE_STYLES[idx % len(LINE_STYLES)], linewidth=LINE_WIDTH, 
                    marker=MARKERS[idx%len(MARKERS)])
        ax.fill_between(range(max_len), mean_res - std_res, mean_res + std_res, color=colors[name], alpha=0.1)    
        
        # # 2. Average Residual vs Time  (now padded, averaged, and with std bands!)
        # max_len = max(len(res) for res in histories[name]["res"])
        # padded_res = np.array([np.pad(res, (0, max_len - len(res)), constant_values=res[-1]) for res in histories[name]["res"]])
        # padded_times = np.array([np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) for t in histories[name]["time"]])
        
        # mean_time = np.mean(padded_times, axis=0)
        # mean_res = np.mean(padded_res, axis=0)
        # std_res = np.std(padded_res, axis=0)
        
        # ax[1].semilogy(mean_time, mean_res, label=name, color=colors[name],
        #             linestyle=LINE_STYLES[idx % len(LINE_STYLES)], linewidth=LINE_WIDTH,
        #             marker=MARKERS[idx%len(MARKERS)])
        # ax[1].fill_between(mean_time, mean_res - std_res, mean_res + std_res, color=colors[name], alpha=0.1)
    # ============================RESIDUALS===========================

    # Title (underscores escaped for proper rendering)
    # title_str = dataset_name.replace('_', r'\_')
    # ax.set_title(f'{title_str}-Convergence', fontsize=TITLE_FONT_SIZE, fontweight='bold')

    fig.tight_layout()

    ax.set_xlabel("OuIter")
    ax.set_ylabel("Relative Residual")
    ax.set_title(f"Residual vs OuIter" )# (Avg over ({NUM_TRIALS}) Seeds)")
    ax.set_xlim(left=0)
    ax.tick_params(axis='both', labelsize=AXES_FONT_SIZE)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.legend(
                fontsize=LEGEND_FONT_SIZE,
                loc='best',
                frameon=True,
                facecolor=(0.95, 0.95, 0.95))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)


    # ax[1].set_xlabel("Time (Seconds)")
    # ax[1].set_ylabel("Relative Residual")
    # ax[1].set_title(f"Residual vs Time") # (Avg over ({NUM_TRIALS}) Seeds)")
    # ax[1].set_xlim(left=0)
    # ax[1].tick_params(axis='both', labelsize=AXES_FONT_SIZE)
    # ax[1].yaxis.set_minor_locator(plt.NullLocator())
    # ax[1].legend(
    #             fontsize=LEGEND_FONT_SIZE,
    #             loc='best',
    #             frameon=True,
    #             facecolor=(0.95, 0.95, 0.95))
    # ax[1].grid(True, linestyle='--', alpha=0.3)
    # ax[1].set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    m,n,r=prob
    # Save
    # output_dir = 'plots_conlogis_new'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{m}_{n}_{r}_convergence.png')
    fig.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close(fig)