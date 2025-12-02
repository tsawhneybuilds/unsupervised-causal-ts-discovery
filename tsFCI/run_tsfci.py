import rpy2.robjects as ro
import networkx as nx
import os

def tsfci(path, sig, tau):
    r_path = 'C:/Users/sah12/Downloads/RCode_TETRADjar_tsFCI/RCode_TETRADjar'

    ro.r(f"setwd('{r_path}')")
    ro.r('options(encoding="latin1")')
    # Source the script
    ro.r("source('start_up.R')")

    start_up = ro.globalenv['start_up']
    start_up()

    graph_path = 'C:/Users/sah12/Downloads/RCode_TETRADjar_tsFCI/RCode_TETRADjar' + '/tempfile.dot'
    if os.path.exists(graph_path):
        print(".dot found")
        os.remove(graph_path)

    ro.r(f'my_data <- read.csv("{path}", stringsAsFactors=FALSE, fileEncoding="latin1")')

    ro.r('tsfci_data <- my_data[, -1]')
    ro.r(f"realData_tsfci(data=tsfci_data, sig={sig}, nrep={tau+1}, makeplot=TRUE)")

    
    graph = nx.nx_pydot.read_dot(graph_path)


    return graph