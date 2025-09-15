import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
list_csv_files=["Exports_goods.csv","External_trades_in_goods.csv","Imports_of_good.csv","Imports_exports_goods_area.csv","Mainland exports.csv"]

class DataAcquisition:
    def __init__(self,file):
        self.file=file
        pass

    def acquire_data(self):
        """
        Acquires and processes data from a CSV file.
        The function remove the wrong value and of the dataset and rename the columns to have me clarity on the content
        """
        dataset = pd.read_csv(self.file, sep=";")
        subset = dataset[["Unnamed: 0", "NOK Million","Unnamed: 2"]]
        subset.columns=["Category","Value (NOK million) - July 2024","Value (NOK million) - July 2025"]
        subset["Value (NOK million) - July 2024"]=pd.to_numeric(subset["Value (NOK million) - July 2024"], errors="coerce")
        subset = subset.dropna()
        # Store the clean data array as a class attribute
        self.data_array=subset.to_numpy()
        self.values=self.data_array[:, 1].astype(float)

    def bar_plot(self):
        """
        Make a bar plot of the top 10 categories based on the values in the second column.
        The x-axis represents the categories (from the first column) and the y-axis represents the values (from the second column).
        I choose to represent the top 10 categories because representing the 100 categories would make the plot unreadable.
        """
        category = self.data_array[:, 0]   # années (probablement en string)
        top10_idx = np.argsort(self.values)[-10:]   # indices des 10 plus grands
        top10_categories = category[top10_idx]
        top10_values = self.values[top10_idx]

        plt.bar(top10_categories, top10_values, color="orange")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Exportations (NOK Million)")
        plt.title("Top 10 of exportations categories")
        plt.tight_layout()
        plt.show()

    def histogram_plot(self):
        """
        Make a histogram plot of the values in the second column.
        The x-axis represents the values (from the second column) and the y-axis represents the frequency of these values.
        """
        values = self.data_array[:, 1].astype(float)  # exportations en NOK million (converti en float)

        plt.hist(self.values, bins=20,color="blue", edgecolor="black")
        plt.xlabel("Exportations (NOK Million)")
        plt.ylabel("Frequency")
        plt.title("Histogramme des exportations")
        plt.tight_layout()
        plt.show()

    def PMF_point_plot(self):
        """
        Make a PMF plot of the values in the second column.
        The x-axis represents the values (from the second column) and the y-axis represents the probability of these values.
        """
        counts, bin_edges = np.histogram(self.values, bins=20, density=True)
        pmf = counts / counts.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.plot(bin_centers, pmf, marker="o", linestyle="none", color="green")
        plt.xlabel("Exportations (NOK Million)")
        plt.ylabel("Probability")
        plt.title("PMF des exportations")
        plt.tight_layout()
        plt.show()

    def PMF_bar_plot(self):
        """
        PMF with bar values, wich is going to be very similar from the histogram plot, but with a better scaling. 
        """
        counts, self.bins = np.histogram(self.values, bins=10)
        self.pmf = counts / counts.sum()
        # Print to see the number of counts and the PMF self.values
        print("Counts :", counts)
        print("PMF :", self.pmf)
        plt.bar(self.bins[:-1], self.pmf, width=np.diff(self.bins), edgecolor="black", align="edge")
        plt.xlabel("Valeurs (NOK Million)")
        plt.ylabel("Probabilité")
        plt.title("PMF des exportations")
        plt.show()


    def CMF_plot(self):
        """
        Make a Cumulative mass function plot, based on the value of the PMF bar plot.
        The first step is to calculate the cumulative sum of the PMF values using numpy's cumsum function.
        After we plot witha  barchart the CDF values against the bin edges.
        """
        cdf = np.cumsum(self.pmf)
        plt.bar(self.bins[:-1], cdf, width=np.diff(self.bins), edgecolor="black", align="edge", color="lightgreen")
        plt.xlabel("Valeurs (NOK Million)")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF des exportations (bar chart)")
        plt.show()


    