import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import imageio.v2 as imageio
from scipy.stats import linregress
import shutil

#------------------------------
# ANIMATION PLOT
#------------------------------
def animate(networks,pos_type="circular",path="output.gif",plot_every=1):
    

    # Build output folder for temp pngs
    temp_folder = "temp"
    if(os.path.exists(temp_folder)):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)

    # POSITIONS LAYOUT
    if(pos_type=="spring"):
            pos=nx.spring_layout(networks[-1])
    elif(pos_type=="random"):
            pos=nx.random_layout(networks[-1])
    else:
            pos=nx.circular_layout(networks[-1])  

    k = 1

    for G in networks:
        N=len(G.nodes)
 
        #INITIALIZE FIGURE AND PLOT
        fig, ax = plt.subplots()  
        fig.set_size_inches(5, 5)

        #GET MIN AND MAX POSITION
        tmpx=[]; tmpy=[]
        for i in pos.keys():
                tmpx.append(pos[i][0])
                tmpy.append(pos[i][1])
        Lxmin=min(tmpx)-0.2; Lxmax=max(tmpx)+0.2
        Lymin=min(tmpy)-0.2; Lymax=max(tmpy)+0.2

        #DRAW BOX
        ax.axhline(y=Lymin)
        ax.axvline(x=Lxmin)
        ax.axhline(y=Lymax)
        ax.axvline(x=Lxmax)

        ## PLOTTING THE NETWORK
        # NODE COLORS
        cmap=plt.colormaps.get_cmap('gnuplot2')

        if(nx.is_directed(G)):
                in_deg_dict = nx.in_degree_centrality(G)
                in_degree = [in_deg_dict[n] for n in G.nodes()]
                out_deg_dict = nx.out_degree_centrality(G)
                out_degree = [out_deg_dict[n] for n in G.nodes()]

                node_sizes = [4000*u/(0.01+max(in_degree)) for u in in_degree]
                node_colors = [cmap(u/(0.01+max(out_degree))) for u in out_degree]
        else:
                deg_dict = nx.degree_centrality(G)
                degree = [deg_dict[n] for n in G.nodes()]

                node_colors = [cmap(u/(0.01+max(degree))) for u in degree]
                node_sizes = [4000*u/(0.01+max(degree)) for u in degree]

        #PLOT NETWORK
        nx.draw(G,
                with_labels=True,
                edgecolors="black",
                node_color=node_colors,
                node_size=node_sizes,
                font_color='lightblue',
                font_size=18,
                pos=pos
                )

        # save the network drawing
        if k < 10:
                node_num = str(0)+str(k)
        else:
                node_num = str(k)
        file_path = temp_folder+"/temp-"+node_num+".png"
        fig.savefig(file_path)
        plt.close(fig) 
        # print(k)
        k = k + 1

    # GIF Stuff
    # list of temp files
    file_names = sorted(os.listdir(temp_folder))
    # animate the images
    images = list(map(lambda filename: imageio.imread(temp_folder+"/"+filename), file_names))
    imageio.mimsave(path, images,fps=3 ) # modify the frame duration as needed
    # display the .gif
    from IPython.display import display,Image
    display(Image(data=open(path,'rb').read(), format='png', width=800))

    # Remove output folder with temp pngs
    shutil.rmtree(temp_folder)

#------------------------------
# NETWORK CENTRALITY CORRELATION PLOTS
#------------------------------
def plot_centrality_correlation(G,path=""):
    if(nx.is_directed(G)):
        # turn the metrics into sequences
        in_degree_sequence = nx.in_degree_centrality(G)
        out_degree_sequence = nx.out_degree_centrality(G)
        in_closeness_sequence = nx.closeness_centrality(G)
        # use the reverse of G to get the out closeness
        out_closeness_sequence = nx.closeness_centrality(G.reverse())
        betweenness_sequence = nx.betweenness_centrality(G)
        # combine sequences
        data = {"In Degree": in_degree_sequence, "Out Degree": out_degree_sequence, "In Closeness": in_closeness_sequence,
                "Out Closenss": out_closeness_sequence, "Betweeness": betweenness_sequence}

    else:
        degree_sequence = nx.degree_centrality(G)
        closeness_sequence = nx.closeness_centrality(G)
        betweenness_sequence = nx.betweenness_centrality(G)
        # combine sequences
        data = {"Degree": degree_sequence, "Closeness": closeness_sequence, "Betweenness": betweenness_sequence}

    # turn into a dataframe
    df = pd.DataFrame(data)
    # plot 
    plot = sns.pairplot(df)

    # save if speicified
    if path != "":
        plot.savefig(path)

#------------------------------
# AVERAGE DEGREE
#------------------------------
def ave_degree(G):
    if(nx.is_directed(G)):
        in_degree_sequence = nx.in_degree_centrality(G)
        out_degree_sequence = nx.out_degree_centrality(G)
        avg_in_degree = sum(in_degree_sequence)/(len(in_degree_sequence))
        avg_out_degree = sum(out_degree_sequence)/(len(out_degree_sequence))
        print("Average In Degree:", avg_in_degree)
        print("Average Out Degree:", avg_out_degree)
    else:
        degree_sequence = nx.degree_centrality(G)
        avg_degree = sum(degree_sequence)/(len(degree_sequence))
        print("Average Degree:", avg_degree)


#------------------------------
# PLOT DEGREE DISTRIBUTION
#------------------------------
def plot_degree_distribution(G,type="in",path="",fit=False):
    # init figure
    fig = plt.figure("Network Summary Graphs", figsize=(30,5))

    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(1, 4)

    N = len(G.nodes)

    def bin_values(values, num_bins=int(N/10)):
        
        # create histogram
        counts, bin_edges = np.histogram(values, bins=num_bins)
        
        # calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # calculate probabs
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        probs = counts / (len(values) * bin_widths) 

        cdf = np.cumsum(probs * bin_widths)

        # filter out zero-probability bins
        mask = probs > 0
        bin_centers = bin_centers[mask]
        probs = probs[mask]
        cdf = cdf[mask]
        
        # return values
        return bin_centers, probs, cdf
    

    if(nx.is_directed(G)):
        if type == "out":
            degree_sequence = [deg for _, deg in G.out_degree()]
        else:
            degree_sequence = [deg for _, deg in G.in_degree()]
    else:
        degree_sequence = [deg for _, deg in G.degree()]

    # DEGREE PLOTS
    degree_sequence = sorted(degree_sequence, reverse=True)

    #  compute probs and cdf
    bin_degree, probs_degree, cdf_degree = bin_values(degree_sequence)

    # PLOTS
    # PDF
    ax0 = fig.add_subplot(axgrid[0,0])
    ax0.hist(degree_sequence, bins=30, density = True, color='cadetblue')
    ax0.set_ylabel("Probability")
    if(nx.is_directed(G)):
        if type == "in":
            ax0.set_xlabel("In Degree")
        elif type == "out":
            ax0.set_xlabel("Out Degree")
        else:
            ax0.set_xlabel("Degree")
    else: 
        ax0.set_xlabel("Degree")

    # LOG PDF
    ax1 = fig.add_subplot(axgrid[0,1])
    ax1.scatter(bin_degree, probs_degree, color='cadetblue')
    ax1.set_ylabel("Probability - log")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if(nx.is_directed(G)):
        if type == "in":
            ax1.set_xlabel("In Degree - log")
        elif type == "out":
            ax1.set_xlabel("Out Degree - log")
        else:
            ax1.set_xlabel("Degree - log")
    else:
        ax1.set_xlabel("Degree - log")

    # Add the Linear Regression Fits
    if fit:
        # Remove zero or negative values for log-log regression
        log_x = np.log10(bin_degree)
        log_y = np.log10(probs_degree)

        # Perform linear regression
        slope, intercept, *_= linregress(log_x, log_y)

        # Create fitted line in log-log space
        fit_y = 10**intercept * (bin_degree ** slope)

        # Plot fitted line
        ax1.plot(bin_degree, fit_y, color='darkred', linestyle='--')

    # CDF
    ax2 = fig.add_subplot(axgrid[0,2])
    ax2.step(bin_degree, cdf_degree,  where='post', color='cadetblue')
    ax2.set_ylabel("CDF")
    if(nx.is_directed(G)):
        if type == "in":
            ax2.set_xlabel("In Degree")
        elif type == "out":
            ax2.set_xlabel("Out Degree")
        else:
            ax2.set_xlabel("Degree")
    else:
        ax2.set_xlabel("Degree")

    # LOG LOG CDF
    ax3 = fig.add_subplot(axgrid[0,3])
    ax3.step(bin_degree, cdf_degree,  where='post', color='cadetblue')
    ax3.set_ylabel("CDF - log")
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    if(nx.is_directed(G)):
        if type == "in":
            ax3.set_xlabel("In Degree - log")
        elif type == "out":
            ax3.set_xlabel("Out Degree - log")
        else:
            ax3.set_xlabel("Degree - log")
    else:
        ax3.set_xlabel("Degree - log")

    if fit:
        # Remove zero or negative values for log-log regression
        log_x = np.log10(bin_degree)
        log_y = np.log10(cdf_degree)

        # Perform linear regression
        slope, intercept, *_= linregress(log_x, log_y)

        # Create fitted line in log-log space
        fit_y = 10**intercept * (bin_degree ** slope)

        # Plot fitted line
        ax3.plot(bin_degree, fit_y, color='darkred', linestyle='--')

    plt.show()

    # save to output
    if path != "":
        fig.savefig(path)

#------------------------------
# NETWORK PLOTTING FUNCTION
#------------------------------
def plot_network(G,node_color="degree",layout="random"):
    
    # POSITIONS LAYOUT
    N=len(G.nodes)
    if(layout=="spring"):
        # pos=nx.spring_layout(G,k=50*1./np.sqrt(N),iterations=100)
        pos=nx.spring_layout(G)

    if(layout=="random"):
        pos=nx.random_layout(G)

    #INITALIZE PLOT
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    # NODE COLORS
    cmap=plt.cm.get_cmap('Greens')

    # DEGREE 
    if node_color=="degree":
            centrality=list(dict(nx.degree(G)).values())
  
    # BETWENNESS 
    if node_color=="betweeness":
            centrality=list(dict(nx.betweenness_centrality(G)).values())
  
    # CLOSENESS
    if node_color=="closeness":
            centrality=list(dict(nx.closeness_centrality(G)).values())

    # NODE SIZE CAN COLOR
    node_colors = [cmap(u/(0.01+max(centrality))) for u in centrality]
    node_sizes = [4000*u/(0.01+max(centrality)) for u in centrality]

    # #PLOT NETWORK
    nx.draw(G,
            with_labels=True,
            edgecolors="black",
            node_color=node_colors,
            node_size=node_sizes,
            font_color='black',
            font_size=18,
            pos=pos
            )

    plt.show()

#------------------------------
# NETWORK SUMMARY FUNCTION
#------------------------------
def network_summary(G):

    def centrality_stats(x):
        x1=dict(x)
        x2=np.array(list(x1.values())); #print(x2)
        print("	min:" ,min(x2))
        print("	mean:" ,np.mean(x2))
        print("	median:" ,np.median(x2))
        # print("	mode:" ,stats.mode(x2)[0][0])
        print("	max:" ,max(x2))
        x=dict(x)
        sort_dict=dict(sorted(x1.items(), key=lambda item: item[1],reverse=True))
        print("	top nodes:",list(sort_dict)[0:6])
        print("	          ",list(sort_dict.values())[0:6])

    try: 
        print("GENERAL")
        print("	number of nodes:",len(list(G.nodes)))
        print("	number of edges:",len(list(G.edges)))

        print("	is_directed:", nx.is_directed(G))
        print("	is_weighted:" ,nx.is_weighted(G))


        if(nx.is_directed(G)):
            print("IN-DEGREE (NORMALIZED)")
            centrality_stats(nx.in_degree_centrality(G))
            print("OUT-DEGREE (NORMALIZED)")
            centrality_stats(nx.out_degree_centrality(G))
        else:
            print("	number_connected_components", nx.number_connected_components(G))
            print("	number of triangle: ",len(nx.triangles(G).keys()))
            print("	density:" ,nx.density(G))
            print("	average_clustering coefficient: ", nx.average_clustering(G))
            print("	degree_assortativity_coefficient: ", nx.degree_assortativity_coefficient(G))
            print("	is_tree:" ,nx.is_tree(G))

            if(nx.is_connected(G)):
                print("	diameter:" ,nx.diameter(G))
                print("	radius:" ,nx.radius(G))
                print("	average_shortest_path_length: ", nx.average_shortest_path_length(G))

            #CENTRALITY 
            print("DEGREE (NORMALIZED)")
            centrality_stats(nx.degree_centrality(G))

            print("CLOSENESS CENTRALITY")
            centrality_stats(nx.closeness_centrality(G))

            print("BETWEEN CENTRALITY")
            centrality_stats(nx.betweenness_centrality(G))
    except:
        print("unable to run")

#------------------------------
# ISOLATE GCC
#------------------------------
def isolate_GCC(G):
    comps = sorted(nx.connected_components (G),key=len, reverse=True) 
    nodes_in_giant_comp = comps[0]
    return nx.subgraph(G, nodes_in_giant_comp)

