"""
dlc2kinematics
© M. Mathis Lab
https://github.com/AdaptiveMotorControlLab/dlc2kinematics/
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dlc2kinematics.utils.auxiliaryfunctions import read_config
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class Visualizer3D:
    def __init__(self, project_config, datafile3d, other_files=()):
        """
        Minimalistic 3D data visualizer.

        Parameters
        ----------
        project_config : str
            Full path to a DeepLabCut 3D project config YAML file

        datafile3d : str
            Full path to the h5 file containing the 3D data

        other_files : list of str
            List of paths to the h5 files containing the 2D data.
            The length of the list is equal to the number of cameras.

        Examples
        --------
        viz = Visualizer3D(config, cam_3d, [cam0_2d, cam1_2d])
        viz.view(show_axes=True, show_grid=True)
        """
        self.cfg = read_config(project_config)
        self.filename = datafile3d
        self.df = pd.read_hdf(datafile3d)
        self.data_flat = self.df.values
        self.data = self.data_flat.reshape((self.nframes, -1, 3))
        self.other_data = []
        # Hide uncertain predictions
        mask = np.zeros((self.nframes, self.nbodyparts), dtype=bool)
        for file in other_files:
            df = pd.read_hdf(file)
            data = df.values.reshape((df.shape[0], -1, 3))
            uncertain = data[:, :, 2] < self.cfg["pcutoff"]
            data[uncertain] = np.nan
            mask[uncertain[: self.nframes]] = True
            self.other_data.append(data[:, :, :2])
        self.data[mask] = np.nan

        cmap = plt.cm.get_cmap(self.cfg["colormap"], self.nbodyparts)
        self.colors = cmap(range(self.nbodyparts))

        # Cache skeleton
        bodyparts = self.df.columns.get_level_values("bodyparts").unique().tolist()
        links = [
            (bodyparts.index(bp1), bodyparts.index(bp2))
            for bp1, bp2 in self.cfg["skeleton"]
        ]
        self.ind_links = tuple(zip(*links))

    @property
    def nbodyparts(self):
        return self.data.shape[1]

    @property
    def nframes(self):
        return self.data_flat.shape[0]

    def view(self, figsize=(10, 6), show_axes=False, show_grid=False):
        self.fig = plt.figure(figsize=figsize)
        n_other_data = len(self.other_data)
        gs = self.fig.add_gridspec(n_other_data or 1, 2 if n_other_data else 1)
        self.ax_3d = self.fig.add_subplot(gs[:, -1], projection="3d")
        self.ax_3d.view_init(elev=-90, azim=-90)
        self.ax_2d = [
            self.fig.add_subplot(gs[i, 0]) for i in range(len(self.other_data))
        ]
        plt.subplots_adjust(bottom=0.2)
        for ax in [self.ax_3d] + self.ax_2d:
            if not show_axes:
                ax.axis("off")
            if not show_grid:
                ax.grid(False)

        # Trick to force equal aspect ratio of 3D plots
        minmax = np.c_[
            np.nanmin(self.data, axis=(0, 1)), np.nanmax(self.data, axis=(0, 1))
        ]
        minmax *= 1.1
        minmax_range = (minmax[:, 1] - minmax[:, 0]).max() / 2
        mid_x = np.mean(minmax[0])
        mid_y = np.mean(minmax[1])
        mid_z = np.mean(minmax[2])
        self.ax_3d.set_xlim(mid_x - minmax_range, mid_x + minmax_range)
        self.ax_3d.set_ylim(mid_y - minmax_range, mid_y + minmax_range)
        self.ax_3d.set_zlim(mid_z - minmax_range, mid_z + minmax_range)

        self.points_3d = self.ax_3d.scatter(
            [], [], [], s=self.cfg["dotsize"], alpha=self.cfg["alphaValue"]
        )
        coords = self.data[0]
        self.points_3d._offsets3d = coords.T
        segs = coords[tuple([self.ind_links])].swapaxes(0, 1)
        self.coll_3d = Line3DCollection(segs, colors=self.cfg["skeleton_color"])
        self.ax_3d.add_collection(self.coll_3d)

        # 2D plots data
        self.points_2d = []
        self.coll_2d = []
        for n, ax in enumerate(self.ax_2d):
            lim = np.c_[
                np.nanmin(self.other_data[n], axis=(0, 1)),
                np.nanmax(self.other_data[n], axis=(0, 1)),
            ]
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])
            ax.invert_yaxis()
            data = self.other_data[n][0]
            self.points_2d.append(
                ax.scatter(
                    *data.T,
                    s=self.cfg["dotsize"],
                    alpha=self.cfg["alphaValue"],
                    c=self.colors
                )
            )
            segs = data[tuple([self.ind_links])].swapaxes(0, 1)
            coll = LineCollection(segs, colors=self.cfg["skeleton_color"])
            ax.add_collection(coll)
            self.coll_2d.append(coll)

        ax_slider = self.fig.add_axes([0.2, 0.1, 0.6, 0.03], facecolor="lightblue")
        self.slider = Slider(
            ax_slider, "", 1, self.nframes - 1, valinit=1, valfmt="%1.0f"
        )
        self.slider.on_changed(self.on_change)

    def update(self, i):
        coords = self.data[i]
        self.points_3d._offsets3d = coords.T
        segs = coords[tuple([self.ind_links])].swapaxes(0, 1)
        self.coll_3d.set_segments(segs)
        for data, points, coll in zip(self.other_data, self.points_2d, self.coll_2d):
            points.set_offsets(data[i])
            coll.set_segments(data[i][tuple([self.ind_links])].swapaxes(0, 1))

    def on_change(self, val):
        self.update(int(val))


class MinimalVisualizer3D:
    def __init__(self, data3d, ind_links=()):
        self.data_flat = data3d
        self.data = self.data_flat.reshape((self.nframes, -1, 3))
        self.ind_links = ind_links

    @property
    def nframes(self):
        return self.data_flat.shape[0]

    def add_to_fig(self, fig, loc, show_axes, show_grid):
        self.ax_3d = fig.add_subplot(loc, projection="3d")
        self.ax_3d.view_init(elev=-90, azim=-90)
        plt.subplots_adjust(bottom=0.2)
        if not show_axes:
            self.ax_3d.axis("off")
        if not show_grid:
            self.ax_3d.grid(False)

        # Trick to force equal aspect ratio of 3D plots
        minmax = np.c_[
            np.nanmin(self.data, axis=(0, 1)), np.nanmax(self.data, axis=(0, 1))
        ]
        minmax *= 1.1
        minmax_range = (minmax[:, 1] - minmax[:, 0]).max() / 2
        mid_x = np.mean(minmax[0])
        mid_y = np.mean(minmax[1])
        mid_z = np.mean(minmax[2])
        self.ax_3d.set_xlim(mid_x - minmax_range, mid_x + minmax_range)
        self.ax_3d.set_ylim(mid_y - minmax_range, mid_y + minmax_range)
        self.ax_3d.set_zlim(mid_z - minmax_range, mid_z + minmax_range)

        self.points_3d = self.ax_3d.scatter([], [], [], s=6, alpha=0.7)
        coords = self.data[0]
        self.points_3d._offsets3d = coords.T
        if self.ind_links:
            segs = coords[tuple([self.ind_links])].swapaxes(0, 1)
            self.coll_3d = Line3DCollection(segs, colors="k")
            self.ax_3d.add_collection(self.coll_3d)

    def view(self, figsize=(10, 6), show_axes=False, show_grid=False):
        self.fig = plt.figure(figsize=figsize)
        self.add_to_fig(self.fig, 111, show_axes, show_grid)
        ax_slider = self.fig.add_axes([0.2, 0.1, 0.6, 0.03], facecolor="lightblue")
        self.slider = Slider(
            ax_slider, "", 1, self.nframes - 1, valinit=1, valfmt="%1.0f"
        )
        self.slider.on_changed(self.on_change)

    def update(self, i):
        coords = self.data[i]
        self.points_3d._offsets3d = coords.T
        if self.ind_links:
            segs = coords[tuple([self.ind_links])].swapaxes(0, 1)
            self.coll_3d.set_segments(segs)

    def on_change(self, val):
        self.update(int(val))


class MultiVisualizer:
    def __init__(self, vizs):
        self.vizs = vizs
        self.n_vizs = len(vizs)
        self.nframes = min(viz.nframes for viz in vizs)

    def populate_window(self, layout=None, show_axes=False, show_grid=False):
        if layout is None:
            grid = 111 + 10 * (self.n_vizs - 1)
            layout = [grid + n for n in range(self.n_vizs)]
        self.fig = plt.figure()
        for viz, loc in zip(self.vizs, layout):
            viz.add_to_fig(self.fig, loc, show_axes, show_grid)
        self.add_widgets()
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)

    def add_widgets(self):
        ax_slider = self.fig.add_axes([0.2, 0.1, 0.6, 0.03], facecolor="lightblue")
        self.slider = Slider(
            ax_slider, "", 1, self.vizs[0].data.shape[0] - 1, valinit=1, valfmt="%1.0f"
        )
        self.slider.on_changed(self.on_change)
        ax_btn_prev = self.fig.add_axes([0.85, 0.1, 0.05, 0.03], facecolor="lightblue")
        ax_btn_next = self.fig.add_axes([0.9, 0.1, 0.05, 0.03], facecolor="lightblue")
        self.btn_prev = Button(ax=ax_btn_prev, label="Prev", hovercolor="tomato")
        self.btn_next = Button(ax=ax_btn_next, label="Next", hovercolor="tomato")
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)

    def next_frame(self, *args):
        val = self.slider.val
        self.slider.set_val(min(val + 1, self.nframes - 1))

    def prev_frame(self, *args):
        val = self.slider.val
        self.slider.set_val(max(val - 1, 1))

    def update(self, i):
        for viz in self.vizs:
            viz.update(i)

    def on_change(self, val):
        self.update(int(val))

    def on_move(self, event):
        ax_fig_only = self.fig.axes[: self.n_vizs]  # Ignore all but the visualizer axes
        mask = np.array([event.inaxes == ax for ax in ax_fig_only])
        if mask.any():
            ax_event = self.fig.axes[np.flatnonzero(mask)[0]]
            ax_others = [
                ax for n, ax in enumerate(self.fig.axes) if n in np.flatnonzero(~mask)
            ]
            # Ugly exception handling but deal with mix of 2D/3D views
            try:
                if ax_event.button_pressed in ax_event._rotate_btn:
                    for ax in ax_others:
                        try:
                            ax.view_init(elev=ax_event.elev, azim=ax_event.azim)
                        except AttributeError:
                            pass
            except AttributeError:
                pass

    def on_press(self, event):
        if event.key == "right":
            self.next_frame()
        elif event.key == "left":
            self.prev_frame()


class Visualizer2D:
    def __init__(self, project_config, h5_2d_file, form_skeleton=True):
        """
        Minimalistic 2D data visualizer.

        Parameters
        ----------
        project_config : str
            Full path to a DeepLabCut 2D project config YAML file

        h5_2d_file : str
            Path to a h5 file containing the 2D data.

        form_skeleton : bool
            By default, display the skeleton.

        Examples
        --------
        viz = Visualizer2D(config, h5_2d [h5_2d_file])
        viz.view(show_axes=True, show_grid=True, show_labels=False)
        """
        self.cfg = read_config(project_config)
        self.filename = h5_2d_file
        self.df = pd.read_hdf(h5_2d_file)
        self.data_flat = self.df.values
        self.data = self.data_flat.reshape((self.nframes, -1, 3))

        # Hide uncertain predictions
        with np.errstate(invalid="ignore"):
            uncertain = self.prob < self.cfg["pcutoff"]
        mask = np.broadcast_to(np.expand_dims(uncertain, axis=2), self.data.shape)
        self.data = np.ma.masked_where(mask, self.data)

        cmap = plt.cm.get_cmap(self.cfg["colormap"], self.nbodyparts)
        self.colors = cmap(range(self.nbodyparts))

        # Cache skeleton
        self.bodyparts = self.df.columns.get_level_values("bodyparts").unique().tolist()
        if form_skeleton:
            links = [
                (self.bodyparts.index(bp1), self.bodyparts.index(bp2))
                for bp1, bp2 in self.cfg["skeleton"]
            ]
        else:
            links = []
        self.ind_links = tuple(zip(*links))

    @property
    def nbodyparts(self):
        return self.data.shape[1]

    @property
    def nframes(self):
        return self.data_flat.shape[0]

    @property
    def xy(self):
        return self.data[:, :, :2]

    @property
    def prob(self):
        return self.data[:, :, 2]

    def add_to_fig(self, fig, loc, show_axes, show_grid, show_labels):
        self.ax = fig.add_subplot(loc)
        self.ax.set_aspect("equal")
        plt.subplots_adjust(bottom=0.2)
        if not show_axes:
            self.ax.axis("off")
        if not show_grid:
            self.ax.grid(False)

        # 2D plots data
        lim = np.c_[
            np.nanmin(self.xy, axis=(0, 1)),
            np.nanmax(self.xy, axis=(0, 1)),
        ]
        self.ax.set_xlim(lim[0])
        self.ax.set_ylim(lim[1])
        self.ax.invert_yaxis()
        colors = self.colors.copy()
        colors[self.xy[0].mask[:, 0]] = np.nan
        self.points = self.ax.scatter(
            *self.xy[0].T,
            s=self.cfg["dotsize"],
            alpha=self.cfg.get("alphavalue", 0.7),
            c=colors
        )
        if self.ind_links:
            segs = self.xy[0][tuple([self.ind_links])].swapaxes(0, 1)
            self.coll = LineCollection(segs, colors=self.cfg["skeleton_color"])
            self.ax.add_collection(self.coll)

        self.labels = []
        if show_labels:
            for bp in self.bodyparts:
                self.labels.append(self.ax.text(None, None, bp))
            self.refresh_labels(self.xy[0])

    def view(
        self, figsize=(10, 6), show_axes=False, show_grid=False, show_labels=False
    ):
        self.fig = plt.figure(figsize=figsize)
        self.add_to_fig(self.fig, 111, show_axes, show_grid, show_labels)
        self.add_widgets()

    def add_widgets(self):
        ax_slider = self.fig.add_axes([0.2, 0.1, 0.6, 0.03], facecolor="lightblue")
        self.slider = Slider(
            ax_slider, "", 1, self.nframes - 1, valinit=1, valfmt="%1.0f"
        )
        self.slider.on_changed(self.on_change)

        ax_btn_prev = self.fig.add_axes([0.85, 0.1, 0.05, 0.03], facecolor="lightblue")
        ax_btn_next = self.fig.add_axes([0.9, 0.1, 0.05, 0.03], facecolor="lightblue")
        self.btn_prev = Button(ax=ax_btn_prev, label="Prev", hovercolor="tomato")
        self.btn_next = Button(ax=ax_btn_next, label="Next", hovercolor="tomato")
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)

    def next_frame(self, event):
        val = self.slider.val
        self.slider.set_val(min(val + 1, self.nframes - 1))

    def prev_frame(self, event):
        val = self.slider.val
        self.slider.set_val(max(val - 1, 1))

    def update(self, i):
        coords = self.xy[i]
        colors = self.colors.copy()
        colors[coords.mask[:, 0]] = np.nan
        self.points.set_offsets(coords)
        self.points.set_color(colors)
        if self.ind_links:
            segs = coords[tuple([self.ind_links])].swapaxes(0, 1)
            self.coll.set_segments(segs)
        self.refresh_labels(coords)

    def refresh_labels(self, pos):
        for xy, label in zip(pos, self.labels):
            label.set_position(xy)
            label.set_visible(not np.ma.is_masked(xy))

    def on_change(self, val):
        self.update(int(val))
