#!/usr/bin/env python
# coding: utf-8

import fiona;
import geopandas as gpd
import gpd_lite_toolbox as glt
import matplotlib.pyplot as plt
import numpy as np
import overpy
import requests
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans

fiona.supported_drivers

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('aimport', 'gpd_lite_toolbox')
from pathlib import Path
from matplotlib.colors import ListedColormap
import libpysal as lps
import esda
import seaborn as sbn
from libpysal.weights.contiguity import Queen
import mapclassify as mc
from branca.colormap import linear
import folium
from folium.plugins import FastMarkerCluster


def calc_area_id(city_name):
    # Geocoding request via Nominatim
    geolocator = Nominatim(user_agent='city_compare')
    geo_results = geolocator.geocode(city_name, exactly_one=False, limit=3)

    # Searching for relation in result set
    for r in geo_results:
        print(r.address, r.raw.get('osm_type'))
        if r.raw.get('osm_type') == 'relation':
            city = r
            break
    # Calculating area id
    return int(city.raw.get('osm_id')) + 3600000000


def build_query(pois, city_name):
    area_id = calc_area_id(city_name)
    query = f'[out:xml];area({area_id})->.searchArea;('
    for tag_type, tag_value in pois:
        if tag_value != None:
            query += f'''(node["{tag_type}"="{tag_value}"](area.searchArea);
                          way["{tag_type}"="{tag_value}"](area.searchArea);
                          relation["{tag_type}"="{tag_value}"](area.searchArea););'''
        else:
            # query all tag values
            query += f'''(node["{tag_type}"](area.searchArea);
                          way["{tag_type}"](area.searchArea);
                          relation["{tag_type}"](area.searchArea););'''

    query += '); out geom;'
    return query


def get_poi_data(pois, city_name):
    api = overpy.Overpass()
    query = build_query(pois, city_name)
    #     result_ = api.query(query)
    response = requests.get(api.url,
                            params={'data': query})
    result = api.parse_xml(response.content)

    return response, result


def save_response(file, response):
    p = Path(f'./osm')
    p.mkdir(parents=True, exist_ok=True)
    with Path.open(p / file, 'wb') as f:
        print(f'writing osm data to {file} ...')
        f.write(response.content)
        print('Done.')


def osm2geojson(osmfile, geojsonfile):
    Path(f'./geojson').mkdir(parents=True, exist_ok=True)
    cmd = f'./node_modules/osmtogeojson/osmtogeojson osm/{osmfile} > geojson/{geojsonfile}'
    get_ipython().system('{cmd}')


#     args = ['./node_modules/osmtogeojson/osmtogeojson', f'osm/{osmfile}', '>' ,f'geojson/{geojsonfile}']
#     print(args)
# #         return subprocess.run(args,input=osmf,capture_output=True)
#     with open(geojsonfile,'wb') as f: 
#         args = ['./node_modules/osmtogeojson/osmtogeojson', osmfile]
#         subprocess.Popen(args, stdout = f)
#         # subprocess.Popen(['/node_modules/osmtogeojson/osmtogeojson',f'osm/{osmfile}'],shell=True,stdout=f)

#     subprocess.Popen(args)


def get_city_boundary(city_name):
    api = overpy.Overpass()
    query = f'(relation[name=\'{city_name}\'][type=boundary];);out geom qt;'
    print(query)
    response = requests.get(api.url,
                            params={'data': query})
    result = api.parse_xml(response.content)
    return response, result


def create_df(osmfile, geojsonfile, response=-1):
    if response != -1:
        save_response(osmfile, response)
        osm2geojson(osmfile, geojsonfile)
    return gpd.read_file(f'geojson/{geojsonfile}')


def make_transparent(cmap, row=0):
    tr_cmap = cmap(np.arange(cmap.N))  # Get the colormap colors
    tr_cmap[row, -1] = 0  # Set alpha
    return ListedColormap(tr_cmap)  # Create new colormap


def make_folium_grid_map(grid_df, city_boundary_df, col='value'):
    m = folium.Map([40.4406, -79.9959], zoom_start=12)

    if col == 'value':
        colormap = linear.OrRd_06.scale(
            grid_df[col].min(),
            grid_df[col].max())
        colormap.caption = 'value color scale'
        colormap.add_to(m)

        def style_function(x):
            return {
                #             'fillColor': color,
                'fillColor': 'None' if x['properties'][col] == 0 else colormap(x['properties'][col]),
                'color': 'white',
                'weight': 1,
                'dashArray': '10, 10',
                'fillOpacity': .9,
            }
    elif col == 'hotspot':
        def style_function(x):
            # print(x)
            return {
                #             'fillColor': color,
                'fillColor': 'None' if x['properties'][col] == 'hot spot' else 'Red',
                'color': 'Red',
                'weight': 1,
                'dashArray': '10, 10',
                'fillOpacity': .6,
            }



    elif col == 'ylagq5yb':
        colormap = linear.OrRd_06.scale(
            grid_df[col].min(),
            grid_df[col].max())
        colormap.caption = 'value-lag color scale'
        colormap.add_to(m)

        def style_function(x):
            return {
                #             'fillColor': color,
                'fillColor': 'None' if x['properties'][col] == 0 else colormap(x['properties'][col]),
                'color': 'white',
                'weight': 1,
                'dashArray': '10, 10',
                'fillOpacity': .9,
            }

    folium.GeoJson(grid_df,
                   style_function, name='Heat Grid',
                   ).add_to(m)

    folium.GeoJson(city_boundary_df, name='Pittsburgh Boundary',
                   style_function=lambda x: {'color': 'black'}
                   ).add_to(m)

    # m.save(os.path.join('results', 'geopandas_0.html'))
    folium.LayerControl().add_to(m)

    return m


def make_marker_cluster(df, city_boundary_df):
    callback = """\
    function (row) {
        var icon, marker;
        icon = L.AwesomeMarkers.icon({
            icon: "map-marker", markerColor: "red"});
        marker = L.marker(new L.LatLng(row[0], row[1]));
        marker.setIcon(icon);
        return marker;
    };
    """
    # df.to_crs(epsg=4362,inplace=True)
    # Create a Map instance
    m = folium.Map(location=[40.4406, -79.9959],
                   zoom_start=12, control_scale=True)
    data = [(p.y, p.x) for p in df.centroid]
    # print(data)

    FastMarkerCluster(
        data=data,
        callback=callback,
        name='POIs'
    ).add_to(m)

    folium.GeoJson(city_boundary_df, name='Pittsburgh Boundary',
                   style_function=lambda x: {'color': 'black'}
                   ).add_to(m)
    folium.LayerControl().add_to(m)

    # m.save('pgh_marker_clusters.html')
    return m


def plot_cluster_folium(pois, osmfile, geojsonfile, run_query=True):
    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    if run_query:
        response_pois, result_pois = get_poi_data(pois, city_name='Pittsburgh')
        pois_df = create_df(osmfile, geojsonfile, response_pois)
    else:
        pois_df = create_df(osmfile, geojsonfile, response=-1)
    m = make_marker_cluster(pois_df, city_boundary_df)
    # project to UTM zone 17T
    pois_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    m.save('folium_maps/poi_cluster.html')
    return pois_df, m, city_boundary_df


def plot_cluster(pois, osmfile, geojsonfile, run_query=True):
    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    if run_query:
        response_pois, result_pois = get_poi_data(pois, city_name='Pittsburgh')
        pois_df = create_df(osmfile, geojsonfile, response_pois)
    else:
        pois_df = create_df(osmfile, geojsonfile, response=-1)
    # project to UTM zone 17T
    pois_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    city_plot = city_boundary_df.plot(color='grey', figsize=(15, 15))
    pois_plot = pois_df.plot(ax=city_plot, color='blue', markersize=2)
    pois_plot.legend([x[1] for x in pois])
    return pois_df, pois_plot, city_boundary_df


def plot_grid_folium(pois_df, height, use_area):
    # account for areas as linestring
    pois_df.loc[pois_df.geometry.type == 'LineString', 'geometry'] = pois_df.loc[
        pois_df.geometry.type == 'LineString'].convex_hull
    pois_df_centers = pois_df.copy()
    pois_df_centers["geometry"] = pois_df.centroid
    # one count for every poi
    if use_area:
        pois_df_centers['value'] = pois_df.area / (height ** 2)
        categorical = False
    else:
        pois_df_centers['value'] = 1
        categorical = True
        # print(pois_df.crs)
    grid_df = glt.gridify_data(pois_df_centers, height=height, col_name='value', method=np.sum, cut=False)
    grid_df.value.replace(-1, 0, inplace=True)
    grid_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    # Create colormap where color corresponding to minumum value has alpha = 0
    my_cmap = make_transparent(plt.cm.magma_r)
    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    m = make_folium_grid_map(grid_df, city_boundary_df)
    m.save('folium_maps/poi_grid.html')
    return grid_df, m


def plot_grid(pois_df, height, use_area):
    # account for areas as linestring
    pois_df.loc[pois_df.geometry.type == 'LineString', 'geometry'] = pois_df.loc[
        pois_df.geometry.type == 'LineString'].convex_hull
    pois_df_centers = pois_df.copy()
    pois_df_centers["geometry"] = pois_df.centroid
    # one count for every poi
    if use_area:
        pois_df_centers["value"] = pois_df.area / (height ** 2)
        categorical = False
    else:
        pois_df_centers["value"] = 1
        categorical = True
    print(pois_df.crs)
    grid_df = glt.gridify_data(pois_df_centers, height=height, col_name='value', method=np.sum, cut=False)
    grid_df.value.replace(-1, 0, inplace=True)
    grid_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    # Create colormap where color corresponding to minumum value has alpha = 0
    my_cmap = make_transparent(plt.cm.magma_r)
    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    city_plot = city_boundary_df.plot(figsize=(15, 15), color='grey')

    grid_df.plot(ax=city_plot, column='value', cmap=my_cmap, vmin=0, k=5, edgecolor='white', linewidth=0.1, legend=True,
                 categorical=categorical)
    return grid_df


def k_means_analysis(pois_df, max_clusters):
    pos_arr = np.asarray(list(zip(pois1_df_utm_grid.centroid.x, pois1_df_utm_grid.centroid.y)))
    kmeans_arr = []
    for c in range(2, max_clusters + 1):
        kmeans_arr.append(KMeans(n_clusters=c, random_state=0).fit(pos_arr))
    kmeans_scores = [km.inertia_ for km in kmeans_arr]
    plt.plot(range(2, max_clusters + 1), kmeans_scores, '-o')
    return kmeans_arr


def k_means_plot(pois_df, n_clusters=3):
    f, ax = plt.subplots(1, 1, figsize=(15, 15))
    pos_arr = np.asarray(list(zip(pois_df.centroid.x, pois_df.centroid.y)))
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(pos_arr)
    centroids = km.cluster_centers_
    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    city_boundary_df.plot(ax=ax, color='grey')
    ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c=labels, s=4)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
    return labels, centroids


def spatial_lag_folium(df):
    wq = Queen.from_dataframe(df)
    wq.transform = 'r'
    y = df["value"]
    ylag = lps.weights.lag_spatial(wq, y)
    ylagq5 = mc.Quantiles(ylag, k=5)
    #     ylagq5.bins
    # my_cmap = make_transparent(plt.cm.magma_r)
    # f, ax = plt.subplots(1, figsize=(15, 15))
    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    # city_boundary_df.plot(ax=ax,color='grey')
    df = df.assign(ylagq5yb=ylagq5.yb)
    # df.plot(column='ylagq5yb', categorical=True,k=2, cmap=my_cmap, linewidth=0.1, ax=ax,edgecolor='white', legend=True)
    # ax.set_axis_off()
    # plt.title("Spatial Lag Number of POIs (Quintiles)")
    # plt.show()
    m = make_folium_grid_map(df, city_boundary_df, col='ylagq5yb')
    m.save('folium_maps/ylag.html')
    return df, m


def spatial_lag(df):
    wq = Queen.from_dataframe(df)
    wq.transform = 'r'
    y = df["value"]
    ylag = lps.weights.lag_spatial(wq, y)
    ylagq5 = mc.Quantiles(ylag, k=5)
    #     ylagq5.bins
    my_cmap = make_transparent(plt.cm.magma_r)
    f, ax = plt.subplots(1, figsize=(15, 15))
    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    city_boundary_df.plot(ax=ax, color='grey')
    df = df.assign(ylagq5yb=ylagq5.yb)
    df.plot(column='ylagq5yb', categorical=True, k=2, cmap=my_cmap, linewidth=0.1, ax=ax, edgecolor='white',
            legend=True)
    ax.set_axis_off()
    plt.title("Spatial Lag Number of POIs (Quintiles)")
    plt.show()
    return df


def morans_I(df):
    wq = Queen.from_dataframe(df)
    wq.transform = 'r'
    y = df["value"]
    np.random.seed(12345)
    mi = esda.moran.Moran(y, wq)
    print(f'Moran\'s I Coefficient = {mi.I}')
    print(f"""p_sim = {mi.p_sim}
          This is the p-value based on permutations (one-tailed) null: spatial randomness alternative: the 
          observed I is extreme if it is either extremely greater or extremely lower than 
          the values obtained based on permutations""")
    sbn.kdeplot(mi.sim, shade=True)
    plt.vlines(mi.I, 0, 1, color='r')
    plt.vlines(mi.EI, 0, 1)
    plt.xlabel("Moran's I")
    plt.show()
    return mi


def moran_scatterplot(df):
    wq = Queen.from_dataframe(df)
    wq.transform = 'r'
    y = df["value"]
    ylag = lps.weights.lag_spatial(wq, y)
    b, a = np.polyfit(y, ylag, 1)
    f, ax = plt.subplots(1, figsize=(9, 9))
    ax.plot(y, ylag, '.', color='firebrick')
    # dashed vert at mean of the price
    ax.vlines(y.mean(), ylag.min(), ylag.max(), linestyle='--')
    # dashed horizontal at mean of lagged price
    ax.hlines(ylag.mean(), y.min(), y.max(), linestyle='--')
    # red line of best fit using global I as slope
    ax.plot(y, a + b * y, 'r')
    ax.set_title('Moran Scatterplot')
    ax.set_ylabel('Spatial Lag of Value')
    ax.set_xlabel('POI Count')
    # plt.show()
    return f, ax


def find_hotspot_folium(df, threshold):
    wq = Queen.from_dataframe(df)
    wq.transform = 'r'
    y = df["value"]
    li = esda.moran.Moran_Local(y, wq)
    sig = li.p_sim < threshold
    hotspot = sig * li.q == 1  # first quadrant is for hotspots
    spots = ['n.sig.', 'hot spot']
    labels = [spots[i] for i in hotspot * 1]
    # f, ax = plt.subplots(1, figsize=(15, 15))

    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    # city_boundary_df.plot(ax=ax,color='grey')
    df = df.assign(hotspot=labels)

    m = make_folium_grid_map(df, city_boundary_df, col='hotspot')
    m.save('folium_maps/hotspots.html')

    return df, m


def find_hotspots(df, threshold):
    wq = Queen.from_dataframe(df)
    wq.transform = 'r'
    y = df["value"]
    li = esda.moran.Moran_Local(y, wq)
    sig = li.p_sim < threshold
    hotspot = sig * li.q == 1  # first quadrant is for hotspots
    spots = ['n.sig.', 'hot spot']
    labels = [spots[i] for i in hotspot * 1]
    f, ax = plt.subplots(1, figsize=(15, 15))

    city_boundary_df = gpd.read_file('geojson/Pittsburgh_City_Boundary.geojson')
    city_boundary_df.to_crs('+proj=utm +zone=17T +ellps=WGS84 +datum=WGS84 +units=m +no_defs', inplace=True)
    city_boundary_df.plot(ax=ax, color='grey')
    my_cmap = make_transparent(plt.cm.seismic_r, -1)
    df = df.assign(hotspot=labels)
    df.plot(column='hotspot', categorical=True, k=2, cmap=my_cmap, linewidth=0.1, ax=ax, edgecolor='white', legend=True)
    ax.set_axis_off()
    plt.show()
    return df


def analyze(pois, filename, height, run_query=True, use_area=False):
    pois_df, pois_plot, city_boundary_df = plot_cluster(pois, f'{filename}.osm', f'{filename}.geojson',
                                                        run_query=run_query)
    grid_df, m = plot_grid(pois_df, height=height, use_area=use_area)
    spacial_lag_df = spatial_lag(grid_df)
    mi = morans_I(grid_df)
    moran_scatterplot(grid_df)
    hotspot_df = find_hotspots(grid_df)
    return pois_df, (grid_df, m), spacial_lag_df, mi, hotspot_df


def analyze_folium(pois, filename, height, threshold, run_query=True, use_area=False):
    pois_df, m0, city_boundary_df = plot_cluster_folium(pois, f'{filename}.osm', f'{filename}.geojson',
                                                        run_query=run_query)
    grid_df, m1 = plot_grid_folium(pois_df, height=height, use_area=use_area)
    spacial_lag_df, m2 = spatial_lag_folium(grid_df)
    mi = morans_I(grid_df)
    f_moran, ax_moran = moran_scatterplot(grid_df)
    hotspot_df, m3 = find_hotspot_folium(grid_df, threshold)
    return (pois_df, m0), (grid_df, m1), (spacial_lag_df, m2), mi, (f_moran, ax_moran), (hotspot_df, m3)
