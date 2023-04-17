import json
import urllib.request
import csv
import numpy as np
def load_dataset(lags: int = 8):
    
    lags = lags
    url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
    dataset = json.loads(urllib.request.urlopen(url).read())
    timstamps = dataset["time_periods"] 
    headerList = ['timestamp', 'src', 'dst','y','weights']
    stacked_target = np.array(dataset["y"])
    standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
                                        np.std(stacked_target, axis=0) + 10 ** -10)
    print(standardized_target)

    with open("covid19.csv", "w", newline="", encoding="utf-8") as f_output:
        csv_output = csv.writer(f_output, delimiter=',')
        csv_output.writerow(headerList)
        for time in range(1,timstamps):
            count = len(dataset["edge_mapping"]["edge_index"][str(time)])
            for i in range(count):
                timestamp = time
                src = dataset["edge_mapping"]["edge_index"][str(time)][i][0] #src
                dst = dataset["edge_mapping"]["edge_index"][str(time)][i][1] #dest
                weight = dataset["edge_mapping"]["edge_weight"][str(time)][i] #attr/msg
                csv_output.writerow([timestamp, src, dst, standardized_target[time][src], weight, standardized_target[time-1][src], standardized_target[time-1][dst]])
                                    #+ list(dataset["y"][time][region] for region in range(len(dataset["y"][time]))))
            print(count)
load_dataset()