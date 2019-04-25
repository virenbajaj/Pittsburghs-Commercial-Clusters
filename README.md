# Pittsburgh's Commercial Clusters  

This project's aim is to identify centers of commerial activity in the city of Pittsburgh using open GIS data and to create an interactive map to visualize them.

In this notebook I will demonstrate how to identify commercial centers of the city of Pittsburgh, PA using Points of Interest aquired from Open Street Maps.

The first step is to answer the question: what is a commercial center?  
As the name suggests, commercial centers are areas of high economic activity. 
These regions have a high density of businesses that partake in buying and selling of goods and services.
These businesses include retail shops, restaurants, bars, cafes, offices, banks, etc. 
People travel to these commmercial centers on a regular basis to conduct all kinds of economic activity, as a result of which the commercial areas need to encorporate 'features' for the primary mode of transport of the people in the city. 
In Pittsburgh, people primarily travel by car, but it also has well-connected 'bus-driven' public transport system. 


In our analysis we will look at the following major points of interest as indicators for commercial activity:
1. Restaurants
2. Bars
3. Pubs
4. Cafes
5. Shops
6. Offices
7. Commercial Buildings
8. Retail Buildings
9. Bus Stops
10. Parking

Since there is no easy and robust way to validate the actual commercial activity of a region, I have chosen a city of which I have 'expert' knowledge, which is to say, I can confirm/reject an area as being commercially active based on past experience. 

The analysis for the collected POI data can be summarized as follows:
1. Visualize their locations on the map
2. Identify spatial clusters (regions of high density) of that category of POI 
3. Calculate how non-random these clusters are using Moran's I Coefficient of Spatial Autocorrelation 
4. Identify the statistically significant hot-spots of the POIs

Finally, we validate that these are actually commercial centers from expert knowledge.

Improvements: 
1. Calculate Morans's I statistic for each POI and choose only significant POIs
2. Use Land Zoning map and find what percent of hot-spots overlap with commercial land zones. Pitssburgh has, for example, 'Urban Neighborhood Commercial', 'Local Neighborhood Commercial', etc. This can be a metric for goodness of detected hot-spots. 



## Directory Information

1. env0.txt and env1.txt have my current envioronment. They are outputs from pip freeze and conda list respectively.
2. poi_hotspots contains the final hotspot shapefile
3. geojson and osm have geojson and osm files received from either OSM or Western Pennsylvania Regional Data Center (http://www.wprdc.org/) 
4. node_modules is from gpd_lite_toolbox (https://github.com/mthh/gpd_lite_toolbox) used to 'gridify' data
5. Commercial Centers of Pittsburgh is the notebook with the final analysis
6. utils.py has all the helper functions used to conduct the analysis 
7. folium_maps has the html folium maps.

