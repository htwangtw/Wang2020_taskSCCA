import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


from nilearn.plotting import cm

from scipy.stats.mstats import zscore
from scipy.stats import percentileofscore
import seaborn as sns

from src.data_cleaning import clean_confound, mad, select_mad_percentile
from src.utils import unflatten


def rank_labels(pd_ser):
    '''
    rank behaviour variables and ignore labels of sparsed variables.
    return label and a flatten array of the current values
    '''
    pd_ser = pd_ser.replace(to_replace=0, value=np.nan)
    pd_ser = pd_ser.sort_values(ascending=False, )

    behav_labels = list(pd_ser.index)
    v_ranked = pd_ser.values
    v_ranked_flat = np.zeros((len(behav_labels),1))
    v_ranked_flat.flat[:v_ranked.shape[0]] = v_ranked

    return v_ranked_flat, behav_labels

def plot_heatmap(ax, mat, x_labels, y_labels, cb_max, cmap=plt.cm.RdBu_r):
    '''
    plot one single genaric heatmap
    Only when axis is provided

    ax: the axis of figure
    mat: 2-d matrix
    x_labels, y_labels: lists of labels
    cb_max: maxium value of the color bar
    '''
    graph = ax.matshow(mat, vmin=-cb_max, vmax=cb_max, cmap=cmap)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.set_yticklabels(y_labels)
    return graph

def single_heatmap(mat, x_labels, y_labels, cb_label):
    '''
    heat map with color bar
    '''
    cb_max = np.max(np.abs(mat))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    hm = ax.matshow(mat, vmin=-cb_max, vmax=cb_max, cmap=plt.cm.RdBu_r)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=1)
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label(cb_label)

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    return fig

def plot_SCCA_FC_MWQ(FC_ws, behav_ws, region_labels, behav_labels, cb_max, cmap=plt.cm.RdBu_r):
    '''
    plotting tool for functional connectivity vs MRIQ
    '''
    plt.close('all')

    fig = plt.figure(figsize=(15,4))

    ax = fig.add_subplot(111)

    brain = plot_heatmap(ax, FC_ws, region_labels, region_labels, cb_max, cmap)
    # add a line to a diagnal
    ax.plot([-0.5, len(region_labels)-0.5], [-0.5, len(region_labels)-0.5], ls='--', c='.3')

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", size="1%", pad=8)
    behav = plot_heatmap(ax2, behav_ws, [' '], behav_labels, cb_max, cmap)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="50%", pad=0.25)
    fig.colorbar(brain, cax=cax)

    return fig


def map_labels(data, lables):
    df = pd.DataFrame(data, index=lables)
    return df

def show_results(u, v, u_labels, v_labels, rank_v=True, sparse=True):
    '''
    for plotting the scca decompostion heatmapt
    u must be from a functional connectivity data set
    v must be from a data set that can be expressed in a single vector
    '''

    df_v = map_labels(v, v_labels)
    n_component = v.shape[1]

    # find maxmum for the color bar
    u_max = np.max(np.abs(u))
    v_max = np.max(np.abs(v))
    cb_max = np.max((u_max, v_max))

    figs = []

    for i in range(n_component):
    # reconstruct the correlation matrix
        ui = unflatten(u[:, i])

        if rank_v:
            vi, cur_v_labels = rank_labels(df_v.iloc[:, i])


        else:
            vi = v[:, i - 1 :i] # the input of the plot function must be an array
            cur_v_labels = v_labels

        if sparse:
            idx = np.isnan(vi).reshape((vi.shape[0]))
            vi = vi[~idx]
            vi = vi.reshape((vi.shape[0], 1))
            cur_v_labels = np.array(cur_v_labels)[~idx]

        cur_fig = plot_SCCA_FC_MWQ(ui, vi, u_labels, cur_v_labels, cb_max=cb_max, cmap=plt.cm.RdBu_r)
        # save for later
        figs.append(cur_fig)
    return figs

from matplotlib.backends.backend_pdf import PdfPages

def write_pdf(fname, figures):
    '''
    write a list of figures to a single pdf
    '''
    doc = PdfPages(fname)
    for fig in figures:
        fig.savefig(doc, format='pdf', dpi=150, bbox_inches='tight')
    doc.close()

def write_png(fname, figures):
    '''
    write a list of figures to separate png files
    '''
    for i, fig in enumerate(figures):
        fig.savefig(fname.format(i + 1), dpi=150, bbox_inches='tight')

def set_text_size(size):
    '''
    set all the text in the figures
    the font is always sans-serif. You only need this
    '''
    font = {'family' : 'sans-serif',
            'sans-serif' : 'Arial',
            'size' : size}
    matplotlib.rc('font', **font)

class plot_mad_results(object):
    '''
    Dimension reduction on the functional connectivity using
    median absolute distribution.
    '''
    def __init__(self, fc_flatten):
        self.X = fc_flatten
        self.X_mad = mad(fc_flatten)
        self.X_mean = fc_flatten.mean(axis=0)
        self.pr = [50, 75, 90, 95]


    def plot_mad_distroubtion(self, fn):

        x_lim = self.X.shape[1]
        y_lim = self.X_mad.max()

        sort_idx = np.argsort(-self.X_mad)
        X_sort = self.X_mad[sort_idx]


        fig, ax = plt.subplots(figsize=(6,4))


        ax.fill_between(range(0,x_lim), 0, X_sort)

        x_ticks = [x_lim]
        for i in self.pr:
            ax.fill_between(range(0,x_lim), 0, X_sort,
                            where= X_sort > np.percentile(self.X_mad, i))
            t = np.percentile(range(0, x_lim), 100 - i)
            x_ticks.append(int(t))
        x_ticks.append(1)
        x_ticks = sorted(x_ticks)



        plt.xlim((1, x_lim))
        plt.ylim((0, y_lim))
        plt.xticks(x_ticks, rotation=90)

        ax.annotate('95%', xy=(np.percentile(range(0,x_lim), 2), 0.23))
        ax.annotate('90%', xy=(np.percentile(range(0,x_lim), 7), 0.18))
        ax.annotate('75%', xy=(np.percentile(range(0,x_lim), 15), 0.16))
        ax.annotate('50%', xy=(np.percentile(range(0,x_lim), 35), 0.14))

        plt.xlabel('Functional connectivity edges')
        plt.ylabel('Median absolute deviation')
        plt.savefig('reports/figures/{}.png'.format(fn), dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_mad_reduction(self, label_file, fn):
        '''

        requre file generated from PyROICluster
        'references/scorr05_2level_names_100.csv'

        '''

        X_filtered = {'Full matix' : unflatten(self.X_mean)}
        for i in self.pr[:-1]:
            x_f, _ = select_mad_percentile(self.X_mean, self.X_mad, i)
            X_filtered['{}% matix'.format(i)] = unflatten(x_f)

        fig, axarr= plt.subplots(2, 2, figsize=(10, 10))

        for i, (title, mat) in enumerate(X_filtered.items()):
            loc_x, loc_y = np.unravel_index(i, (2,2))
            ax_cur = axarr[loc_x, loc_y]
            ax_cur.set_title(title)

            mat, ticks, ticklabels = sort_by_yeo7(mat, label_file)
            sns.heatmap(mat, center=0,
                        square=True, annot=False,
                        cmap='cold_hot', ax=ax_cur)

            ax_cur.hlines(ticks[1:], *ax_cur.get_xlim(), color='w', lw=0.5)
            ax_cur.vlines(ticks[1:], *ax_cur.get_ylim(), color='w', lw=0.5)

            ax_cur.set_xticks(ticks)
            ax_cur.set_xticklabels(ticklabels)
            ax_cur.set_yticks(ticks)
            ax_cur.set_yticklabels(ticklabels)

        plt.savefig('reports/figures/{}.png'.format(fn), dpi=300, transparent=True)

        plt.show()
        plt.close()



def sort_by_yeo7(mat, label_file):
    '''
    Sort craddock atlas with yeo 7 networks
    require file generated from PyROICluster
    'references/scorr05_2level_names_100.csv'

    mat:
        the n by n matrix to be sorted

    label_file:
        path to label file in .csv generated from PyROICluster
        with Yeo-Krienen 7 networks labels
        The number of rows should match n

    '''

    def get_primary(x):
        list_name = re.findall("[a-zA-Z]+", x)
        if len(list_name) == 1:
            prim = list_name[0]

        elif list_name[0] == 'None':
                prim = list_name[1]
        else:
            prim = list_name[0]
        return prim

    def get_ticks(label_names_yeo7):
        ticks = [1]
        ticklabels = ['Default']
        for i in range(label_names_yeo7.shape[0]):
            cur = label_names_yeo7.iloc[i, 3]
            if ticklabels[-1] != cur:
                ticks.append(i + 1)
                ticklabels.append(cur)
        return ticks

    label_names = pd.read_csv(label_file)

    label_names['Yeo7'] = label_names['Yeo-Krienen 7 networks'].apply(get_primary)
    label_names = label_names.sort_values('Yeo7')
    label_names_yeo7 = label_names.iloc[:, [0, 1, -1]].reset_index()

    tmp = pd.DataFrame(mat,
                    index=range(1, label_names.shape[0] + 1),
                    columns=range(1, label_names.shape[0] + 1))
    idx = label_names_yeo7.index.tolist()
    reorder = tmp.reindex(index=idx, columns=idx)
    ticks = get_ticks(label_names_yeo7)
    ticklabels = ['DMN', 'DAN', 'FPN', 'LIM', 'None', 'S-M', 'VAN', 'VIS']

    return reorder.values, ticks, ticklabels


