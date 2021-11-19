import math
import shapely
import param
import panel as pn
from holoviews import streams
import geopandas as gpd
import geoviews as gv
import pandas as pd
from pydsm.hydroh5 import HydroH5

import datetime

import holoviews as hv
from holoviews import opts
import hvplot.pandas
hv.extension('bokeh')
gv.extension('bokeh')

pn.extension()

# code


def calc_angle(coords):
    c2 = coords[1]
    c1 = coords[0]
    return math.degrees(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))


assert calc_angle([(0, 0), (1, 1)]) == 45


def line(e, n, length, angle):
    # for a size of length 10 and angle 0 as a basic shape
    pa = shapely.geometry.LineString([(0, 0), (10, 0)])
    pa = shapely.affinity.rotate(pa, angle, origin=(0, 0))
    pa = shapely.affinity.scale(pa, length, length, origin=(0, 0))
    pa = shapely.affinity.translate(pa, e, n)
    return pa


def arrow(e, n, angle):
    # for side 10 and equilateral triangle ie. 60/2 = 30 degs
    s = 10
    a = math.radians(30)
    pa = shapely.geometry.Polygon(
        [(2 / 3. * s * math.sin(a), 0),
         (-s / 3 * math.cos(a), -s * math.sin(a)),
         (-s / 3 * math.cos(a), s * math.sin(a))])
    scale = 100
    pa = shapely.affinity.rotate(pa, angle, origin=(0, 0))
    pa = shapely.affinity.scale(pa, scale, scale, origin=(0, 0))
    pa = shapely.affinity.translate(pa, e, n)
    return pa


class InputStage(param.Parameterized):
    base_dir = param.String('.')

    @param.output(('filename_base', param.String), ('filename_alt', param.String))
    def output(self):
        self.filename_base = self.file_selector.value[0]
        self.filename_alt = self.file_selector.value[1]
        return self.filename_base, self.filename_alt


    def panel(self):
        self.file_selector = pn.widgets.FileSelector(directory=self.base_dir)
        return pn.Column(self.file_selector)


class StudyComparisionDashboard(param.Parameterized):
    filename_base = param.String()
    filename_alt = param.String()
    twstr = param.String()
    base_dir = param.String('.')
    # all these are hardwired
    tw_list_file = param.String(f'timeperiods.txt')
    channels_flow_file = param.String(f'channels_flow.txt')
    stage_channels_file = param.String(f'channels_stage.txt')
    dsm2_channels_gis_file = param.String(
        f'v8.2-opendata/gisgridmapv8.2channelstraightlines/dsm2_channels_straightlines_8_2.shp')
    ltrb_bounds = (-121.63, 37.93, -121.27, 37.76)  # south delta bounds
    time_periods = param.List()
    tw_selector = param.Selector()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.view_pane = pn.Column("Building view")

    @param.depends('base_dir', watch=True, on_init=True)
    def _update_base_dir(self):
        self.channels_flow_file = f'{self.base_dir}/channels_flow.txt'
        self.stage_channels_file = f'{self.base_dir}/channels_stage.txt'
        self.dsm2_channels_gis_file = f'{self.base_dir}/v8.2-opendata/gisgridmapv8.2channelstraightlines/dsm2_channels_straightlines_8_2.shp'
        self.tw_list_file = f'{self.base_dir}/timeperiods.txt'
        self.time_periods = self.load_time_periods(self.tw_list_file)

    def load_time_periods(self, time_periods_file):
        time_periods = pd.read_csv(time_periods_file)
        # time_periods
        time_periods['start_time'] = time_periods.iloc[:, 0:3].apply(
            lambda x: datetime.datetime(*x.values), axis="columns")
        time_periods['end_time'] = time_periods.iloc[:, 3:6].apply(
            lambda x: datetime.datetime(*x.values), axis="columns")

        time_periods = time_periods.assign(twstr=time_periods['start_time'].dt.strftime(
            '%d%b%Y').str.upper() + '-' + time_periods['end_time'].dt.strftime('%d%b%Y').str.upper())
        return time_periods['twstr'].to_list()

    def set_map_bounds(self, map):
        # set bounds
        e_lt, n_lt = hv.util.transform.lon_lat_to_easting_northing(
            longitude=self.ltrb_bounds[0], latitude=self.ltrb_bounds[1])
        e_rb, n_rb = hv.util.transform.lon_lat_to_easting_northing(
            longitude=self.ltrb_bounds[2], latitude=self.ltrb_bounds[3])
        return map.opts(xlim=(e_lt, e_rb), ylim=(n_rb, n_lt)).opts(width=900, height=500)

    def create_dsm2_channel_map(self):
        """
        ltrb: left, top, right, bottom longitudes and latitudes
        """
        self.dsm2_channels = gpd.read_file(self.dsm2_channels_gis_file)
        # epsg 3857 for web mercator which is what tiles are based on
        self.dsm2_channels = self.dsm2_channels.to_crs(epsg=3857)
        map = self.dsm2_channels.hvplot(tiles='CartoLight', hover_cols='all')
        # set bounds
        map = self.set_map_bounds(map)
        map = map.opts(opts.Path(line_width=5, line_color="black"))
        return map

    def get_data(hydro, channel_id, location, twstr, vartype):
        if vartype == 'flow':
            data = hydro.get_channel_flow(str(channel_id), location, twstr)
        else:
            data = hydro.get_channel_stage(str(channel_id), location, twstr)
        return data

    def get_mean_flow_data(hydro, twstr, channel_ids):
        return [StudyComparisionDashboard.get_data(hydro, channel_id, 'upstream', twstr, 'flow').mean().values[0] for channel_id in channel_ids]

    def load_stage_channels(self):
        self.stage_channels = pd.read_csv(
            self.stage_channels_file, header=None, names=['channel_id'])
        # stage_channels
        self.stage_channels = self.dsm2_channels[self.dsm2_channels['id'].isin(
            self.stage_channels.channel_id)]
        self.stage_channels = self.stage_channels.reset_index(
            drop=True).rename(columns={'id': 'channel_id'})

    def get_all_stage_data(self, filename_h5):
        hydro = HydroH5(filename_h5)
        location = 'upstream'
        vartype = 'stage'  # 'flow' or 'stage'
        data = [StudyComparisionDashboard.get_data(hydro, channel_id, location, self.twstr, vartype)
                for channel_id in self.stage_channels['channel_id']]

        data = pd.concat(data, axis=1)
        return data

    def load_all_stage_data(self):
        self.load_stage_channels()
        self.hydrow = HydroH5(self.filename_base)
        self.hydrowo = HydroH5(self.filename_alt)
        #location = 'upstream'
        # vartype = 'stage'  # 'flow' or 'stage'
        self.dataw = self.get_all_stage_data(self.filename_base)
        self.datawo = self.get_all_stage_data(self.filename_alt)

    def create_mean_flow_barplot(self):
        flow_channels = pd.read_csv(
            self.channels_flow_file, header=None, names=['channel_id'])
        # flow_channels
        #
        sd_channels = self.dsm2_channels[self.dsm2_channels['id'].isin(
            flow_channels.channel_id)].rename(columns={'id': 'channel_id'})
        sd_channel_map = sd_channels.hvplot(
            tiles='CartoLight', hover_cols='all').opts(opts.Path(line_width=3))
        # arrows
        downstream_end = sd_channels.geometry.apply(lambda x: x.coords[1])
        # downstream_end
        channel_angles = sd_channels.geometry.apply(
            lambda x: calc_angle(x.coords))
        # channel_angles
        arrow_geoms = [arrow(d[0][0], d[0][1], d[1])
                       for d in zip(downstream_end.values, channel_angles.values)]
        arrow_map = gpd.GeoDataFrame(
            geometry=arrow_geoms).hvplot().opts(line_alpha=0)
        # labels
        labelsgdf = gpd.GeoDataFrame(geometry=sd_channels.geometry.centroid).assign(
            text=sd_channels.channel_id.values)
        labelsgdf = labelsgdf.assign(
            x=labelsgdf.geometry.x, y=labelsgdf.geometry.y)
        labelsmap = labelsgdf.hvplot.labels(x='x', y='y', text='text', hover=None).opts(
            opts.Labels(text_font_size='8pt', text_align='left'))
        #
        # sd_channel_map*labelsmap*arrow_map

        flow_mean_values = []
        # for channel_id in flow_channels.iloc[1:2, :].channel_id:
        #    fw = get_data(hydrow, channel_id, 'upstream', twstr, 'flow')
        #    fwo = get_data(hydrowo, channel_id, 'upstream', twstr, 'flow')
        flow_mean_w = StudyComparisionDashboard.get_mean_flow_data(
            self.hydrow, self.twstr, flow_channels.channel_id.to_list())
        flow_mean_wo = StudyComparisionDashboard.get_mean_flow_data(
            self.hydrowo, self.twstr, flow_channels.channel_id.to_list())

        flow_table = flow_channels.assign(
            **{"without": flow_mean_wo, "with": flow_mean_w})

        flows_bar_plot = flow_table.set_index('channel_id').hvplot.bar().opts(
            xrotation=45, title=f'Mean flows with and without barriers: {self.twstr}')

        plot = flows_bar_plot + (sd_channel_map * labelsmap *
                               arrow_map).opts(xaxis=None, yaxis=None)
        plot.cols(1).opts(width=1000)
        return plot

    def show_plot(self, index):
        if index == None or len(index) == 0:
            index = [0]
        tsplotw = self.dataw.iloc[:, index].hvplot(label='with barriers')
        vplotw = self.dataw.iloc[:, index].hvplot.violin()
        tsplotwo = self.datawo.iloc[:, index].hvplot(label='without barriers')
        vplotwo = self.datawo.iloc[:, index].hvplot.violin()
        plot = (tsplotwo * tsplotw).opts(legend_position='top_right') + \
            (vplotwo * vplotw).opts(width=200)
        return plot.opts(title=f'Stage Impact for with and without barriers scenario: {self.twstr}', toolbar=None)

    def create_stage_interactive(self):
        self.stage_channel_map = self.stage_channels.hvplot(
            tiles='CartoLight', hover_cols='all').opts(opts.Path(line_width=5))
        self.stage_channel_map = self.stage_channel_map.opts(
            title='Stage Channels for analysis')
        selection = streams.Selection1D(
            source=self.stage_channel_map.Path.I, index=[0])
        dmap_show_data = hv.DynamicMap(self.show_plot, streams=[selection])
        self.stage_channel_map = self.stage_channel_map.opts(opts.Path(tools=['tap', 'hover'], color='green',
                                                             nonselection_color='blue', nonselection_alpha=0.3,
                                                             frame_width=650))
        self.stage_channel_map = self.set_map_bounds(
            self.stage_channel_map).opts(xaxis=None, yaxis=None, toolbar=None)
        return pn.Column(self.stage_channel_map, dmap_show_data)

    def create_violin_plots(self):
        vopts = opts.Violin(violin_selection_alpha=0, outline_alpha=0,
                            stats_alpha=1, violin_alpha=0.0, box_alpha=0.0, )
        vplotw = self.dataw.hvplot.violin(
            color='red', group_label='with').opts(vopts).opts(xrotation=45)
        vplotw.redim(value='stage').opts(ylabel='stage')
        vplotwo = self.datawo.hvplot.violin(
            color='blue', group_label='without').opts(vopts)
        vplotwo = vplotwo.opts(opts.Violin(violin_fill_alpha=0.1, outline_color='blue',
                                           outline_alpha=0.7)).opts(xrotation=45)
        vplotwo.redim(value='stage').opts(ylabel='stage')
        vplots = vplotw * vplotwo
        vplots = vplots.opts(
            height=400, title=f'Stage Distribution with (red) & without (subdued blue): {self.twstr}')
        return vplots

    def create_box_plots(self):
        from holoviews import dim
        dfwb = self.dataw.rename(columns=dict(zip(self.dataw.columns,
                                                  pd.Series(self.dataw.columns).str.split('-', expand=True).iloc[:, 0])))\
            .melt().assign(scenario='barrier').rename(columns={'variable': 'location', 'value': 'stage'})

        dfnb = self.datawo.rename(columns=dict(zip(self.datawo.columns,
                                                   pd.Series(self.datawo.columns).str.split('-', expand=True).iloc[:, 0])))\
            .melt().assign(scenario='no barrier').rename(columns={'variable': 'location', 'value': 'stage'})

        df = pd.concat([dfwb, dfnb])

        df = df.astype({'location': 'int'}).sort_values(by='location')

        boxwhisker = hv.BoxWhisker(df, ['location', 'scenario'], 'stage').opts(
            width=900, xrotation=45)

        boxwhisker = boxwhisker.opts(show_legend=False, height=400, width=900, cmap='Set1',
                                     box_alpha=0.6, box_fill_color=dim('scenario').str(), whisker_line_dash='dotted')
        return boxwhisker.opts(title=f'Stage with and without barriers: {self.twstr}')

    @param.depends('time_periods', watch=True, on_init=True) 
    def set_time_periods(self):
        self.param.tw_selector.objects = self.time_periods

    @param.depends('tw_selector', 'filename_alt', 'filename_base')
    def view(self):
        if self.tw_selector == None: return 
        self.twstr = self.tw_selector
        self.load_all_stage_data()
        flow_layout = pn.panel(
            self.create_mean_flow_barplot(), linked_axes=False)
        stats_layout = pn.Tabs(('Violin', self.create_violin_plots()),
                            ('Box-Whisker', self.create_box_plots()))
        stage_layout = pn.panel(self.create_stage_interactive(), linked_axes=False)
        self.view_pane = pn.Column(flow_layout, stats_layout, pn.Row('<h2>Interactive Stage Map</h2>'), stage_layout)
        return self.view_pane

    def panel(self):
        self.map = self.create_dsm2_channel_map()
        # all of the below depends upon the tw selected
        layout = pn.Column(pn.Param(self.param.tw_selector), self.view)
        return layout


def create_dashboard(data_dir, **kwargs):

    pipe = pn.pipeline.Pipeline([('Inputs', InputStage(base_dir=data_dir)),
                                ('Dashboard', StudyComparisionDashboard(base_dir=data_dir))], **kwargs)
    # return pipe
    dash_panel = pn.Column(pipe.title, pn.Spacer(), pipe.buttons, pipe.stage)
    return dash_panel.servable()
