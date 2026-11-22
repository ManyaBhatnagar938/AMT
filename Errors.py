from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

def frequencygraph(simulation_folder, save_plot=True):

    simulation_folder = Path(simulation_folder)
    graph_folder = simulation_folder / "Graphs"
    graph_folder.mkdir(exist_ok=True)

    csv_files = sorted(
        f for f in simulation_folder.glob("*.csv")
        if "Raw" not in f.name
    )

    toolbox_keywords = ["basicpitch", "crepe", "librosa", "crepenotes"]

    ground_truth = None
    toolbox_files = []

    for f in csv_files:
        if any(k in f.stem.lower() for k in toolbox_keywords):
            toolbox_files.append(f)
        else:
            ground_truth = f

    def clean_label(file):

        name = file.stem
        name = name.replace("_Formatted", "")
        name = name.replace("_formatted", "")

        if "BasicPitch" in name:
            return "Basic Pitch"

        elif "CrepeNotes" in name:
            return "CREPE Notes"

        elif "CREPE" in name:
            return name.split("Simulation")[-1].replace("_", " ")

        elif "Librosa" in name:
            return name.split("Simulation")[-1].replace("_", " ")

        else:
            return "Ground Truth"

    def filename_label(file):

        name = clean_label(file)
        name = name.replace(" ", "_")
        name = name.replace("/", "_")

        return name

    colour_map = {"Basic Pitch": "tab:blue","CREPE 5ms": "tab:orange","CREPE 10ms": "tab:green","CREPE 15ms": "tab:red","CREPE Notes": "tab:purple","Librosa 5ms": "tab:brown",
    "Librosa 10ms": "tab:pink","Librosa 15ms": "tab:gray"}

    gt = pd.read_csv(ground_truth)
    gt = gt[gt["Frequency (Hz)"].notna() &(gt["Frequency (Hz)"] > 0)]

    for file in toolbox_files:
        label = clean_label(file)
        colour = colour_map.get(label, "tab:blue")
        df = pd.read_csv(file)
        df = df[df["Frequency (Hz)"].notna() &(df["Frequency (Hz)"] > 0)]
        fig, ax = plt.subplots(figsize=(16,7))
        legend_handles = []


        for _, row in gt.iterrows():
            start = row["Start Time (s)"]
            end = row["End Time (s)"]
            freq = row["Frequency (Hz)"]

            ax.hlines(y=freq,xmin=start,xmax=end,color="black",linewidth=1.2,zorder=1)
            ax.scatter(start,freq,marker="D",s=30,color="black",zorder=3)
            ax.scatter(end,freq,marker="D",s=30,color="black",zorder=3)

        legend_handles.append(Line2D([0],[0],color="black",lw=2,marker="D",markersize=5,label="Ground Truth"))

        for _, row in df.iterrows():

            start = row["Start Time (s)"]
            end = row["End Time (s)"]
            freq = row["Frequency (Hz)"]
            ax.hlines(y=freq,xmin=start,xmax=end,color=colour,linewidth=2,alpha=0.9,zorder=2)
            ax.scatter(start,freq,marker="o",color=colour,s=22,zorder=4)
            ax.scatter(end,freq,marker="o",color=colour,s=22,zorder=4)
        legend_handles.append(Line2D([0],[0],color=colour,lw=2,marker="o",markersize=5,label=label))

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.set_title(f"Frequency Comparison - {label}", fontsize=14)

        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))

        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

        ax.grid(which="major", alpha=0.45)
        ax.grid(which="minor", alpha=0.18, linestyle=":")

        ax.legend(handles=legend_handles,loc="upper left")
        plt.tight_layout()

        if save_plot:
            plt.savefig(
                graph_folder / f"Frequency_{filename_label(file)}.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()
        plt.close()

def starttimegraph(simulation_folder, save_plot = "True"):
     
        simulation_folder = Path(simulation_folder)
        graph_folder = simulation_folder / "Graphs"
        graph_folder.mkdir(exist_ok=True)
    
        csv_files = sorted(
            f for f in simulation_folder.glob("*.csv")
            if "Raw" not in f.name
        )
    
        toolbox_keywords = ["basicpitch", "crepe", "librosa", "crepenotes"]

        ground_truth = None
        toolbox_files = []

        for f in csv_files:
            if any(k in f.stem.lower() for k in toolbox_keywords):
                toolbox_files.append(f)
            else:
                ground_truth = f

        def clean_label(file):
            name = file.stem
            name = name.replace("_Formatted", "")
            name = name.replace("_formatted", "")
    
            if "BasicPitch" in name:
                return "Basic Pitch"
    
            elif "CrepeNotes" in name:
                return "CREPE Notes"
    
            elif "CREPE" in name:
                return name.split("Simulation")[-1].replace("_", " ")
    
            elif "Librosa" in name:
                return name.split("Simulation")[-1].replace("_", " ")
    
            else:
                return "Ground Truth"
    
        def filename_label(file):
    
            name = clean_label(file)
            name = name.replace(" ", "_")
            name = name.replace("/", "_")
    
            return name
    
        colour_map = {"Basic Pitch": "tab:blue","CREPE 5ms": "tab:orange","CREPE 10ms": "tab:green","CREPE 15ms": "tab:red","CREPE Notes": "tab:purple","Librosa 5ms": "tab:brown",
        "Librosa 10ms": "tab:pink","Librosa 15ms": "tab:gray"}

        gt = pd.read_csv(ground_truth)
        gt = gt[gt["Frequency (Hz)"].notna() &(gt["Frequency (Hz)"] > 0)]

        onset_groups = {"CREPE": [],"Librosa": [],"Basic Pitch": []}
        for file in toolbox_files:
            label = clean_label(file)
            if "CREPE" in label:
                onset_groups["CREPE"].append(file)

            elif "Librosa" in label:
                onset_groups["Librosa"].append(file)

            elif "Basic Pitch" in label:
                onset_groups["Basic Pitch"].append(file)

        for group_name, files in onset_groups.items():

            if len(files) == 0:
                continue

            fig, ax = plt.subplots(figsize=(15,6))
            legend_handles = []

            for file in files:
                label = clean_label(file)
                colour = colour_map.get(label, "tab:blue")
                df = pd.read_csv(file)
                df = df[df["Frequency (Hz)"].notna() &(df["Frequency (Hz)"] > 0)].reset_index(drop=True)

                j = 0
                x_vals = []
                y_vals = []

                for gt_idx, gt_row in gt.iterrows():
                    gt_midi = int(round(gt_row["MIDI"]))
                    while j < len(df):
                        tb_midi = int(round(df.loc[j, "MIDI"]))
                        if tb_midi == gt_midi:
                            onset_error = (
                                df.loc[j, "Start Time (s)"] -
                                gt_row["Start Time (s)"]
                            )
                            x_vals.append(gt_idx + 1)
                            y_vals.append(onset_error)
                            j = j + 1
                            break
                        j = j + 1

                ax.vlines(x_vals,0,y_vals,color=colour,alpha=0.35,linewidth=1)
                ax.plot(x_vals,y_vals,color=colour,linewidth=1.2,alpha=0.8)
                ax.scatter(x_vals,y_vals,color=colour,s=35,zorder=3)
                legend_handles.append(Line2D([0],[0],marker="o",color=colour,linestyle="-",markersize=6,label=label))

            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Ground Truth Note Number", fontsize=12)
            ax.set_ylabel("Onset Error (s)", fontsize=12)
            ax.set_title(f"Onset Error Comparison - {group_name}", fontsize=14)

            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))

            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.01))

            ax.grid(which="major", alpha=0.45)
            ax.grid(which="minor", alpha=0.18, linestyle=":")

            ax.legend(handles=legend_handles, loc="upper left")
            plt.tight_layout()

            if save_plot:
                plt.savefig(graph_folder / f"Onset_{group_name.replace(' ','_')}.png",dpi=300,bbox_inches="tight")
            plt.show()
            plt.close()

        for file in toolbox_files:
            label = clean_label(file)
            colour = colour_map.get(label, "tab:blue")
            df = pd.read_csv(file)

            df = df[df["Frequency (Hz)"].notna() &(df["Frequency (Hz)"] > 0)]

            fig, ax = plt.subplots(figsize=(16,7))
            legend_handles = []
            ax.scatter(gt["Start Time (s)"],gt["Frequency (Hz)"],marker="D",color="black",s=35,zorder=5)
            legend_handles.append(Line2D([0],[0],marker="D",color="black",linestyle="",markersize=6,label="Ground Truth"))
            ax.scatter(df["Start Time (s)"],df["Frequency (Hz)"],marker="o",color=colour,s=24,zorder=4)
            legend_handles.append(Line2D([0],[0],marker="o",color=colour,linestyle="",markersize=6,label=label))

            ax.set_xlabel("Start Time (s)", fontsize=12)
            ax.set_ylabel("Frequency (Hz)", fontsize=12)
            ax.set_title(f"Start Time vs Frequency - {label}", fontsize=14)

            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.02))

            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(10))

            ax.grid(which="major", alpha=0.45)
            ax.grid(which="minor", alpha=0.18, linestyle=":")

            ax.legend(handles=legend_handles, loc="upper left")
            plt.tight_layout()

            if save_plot:
                plt.savefig(graph_folder / f"StartTime_{filename_label(file)}.png",dpi=300,bbox_inches="tight"                )

            plt.show()
            plt.close()
        
        
frequencygraph("C:\\Users\\PSYLAB03\\Desktop\\Manya\\1-10_Simulations\\Simulation 1a", save_plot =False)
#starttimegraph("C:\\Users\\PSYLAB03\\Desktop\\Manya\\1-10_Simulations\\Simulation 1a")
