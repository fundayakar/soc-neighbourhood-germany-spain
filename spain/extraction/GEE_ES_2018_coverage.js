// Spain SOC replication: source eligibility + 2018 predictor/patch coverage.
// Self-contained. Paste into a NEW GEE script. Creates ONE CSV export task.
// No model fitting. No patch arrays yet. Run the task from the Tasks panel.
var ASSET = 'projects/YOUR_PROJECT/assets/YOUR_LUCAS_2018_ASSET';
var PROJECTION = ee.Projection('EPSG:3035').atScale(20);
var OPTICAL = ['B2','B3','B4','B8','B11','B12'];
var RASTER = OPTICAL.concat(['NDVI','VV','VH','VV_minus_VH',
  'elev','slope','aspect_sin','aspect_cos']);

var points = ee.FeatureCollection(ASSET)
  .filter(ee.Filter.eq('NUTS_0','ES'))
  .filter(ee.Filter.eq('Depth','0-20 cm'))
  .map(function(f) {
    var raw = ee.String(ee.Algorithms.If(
      ee.Algorithms.IsEqual(f.get('OC'),null),'',f.get('OC')))
      .replace('^\\s+|\\s+$','','g');
    var numeric = raw.match('^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$').size().gt(0);
    var oc = ee.Number.parse(ee.String(ee.Algorithms.If(numeric,raw,'-999999')));
    var id = ee.String(f.get('POINTID'));
    var ll = f.geometry().transform('EPSG:4326',0.1).coordinates();
    var xy = f.geometry().transform('EPSG:3035',0.1).coordinates();
    return f.set({point_id:id,source_lon:ll.get(0),source_lat:ll.get(1),
      source_x3035:xy.get(0),source_y3035:xy.get(1),
      eligible:numeric.and(oc.gte(0)).and(id.match('^[0-9]{8}$').size().gt(0))});
  }).filter(ee.Filter.eq('eligible',1))
  .select(['point_id','POINTID','NUTS_0','OC','Depth','SURVEY_DATE',
    'LC','LC0_Desc','LC1_Desc','source_lon','source_lat',
    'source_x3035','source_y3035']);
print('SOURCE COUNT — expected 3763',points.size());
print('UNIQUE IDS — expected 3763',points.aggregate_count_distinct('point_id'));

var region = points.geometry().bounds(100).buffer(2000,100);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region).filterDate('2018-04-01','2018-11-01')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
  .map(function(im) {
    var scl = im.select('SCL');
    var clear = scl.neq(0).and(scl.neq(1)).and(scl.neq(2)).and(scl.neq(3))
      .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
    return im.select(OPTICAL).multiply(0.0001).updateMask(clear);
  });
var reflectance = s2.median();
var ndvi = reflectance.expression('(n-r)/(n+r)',{
  n:reflectance.select('B8'),r:reflectance.select('B4')}).rename('NDVI');
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(region).filterDate('2018-01-01','2019-01-01')
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.eq('orbitProperties_pass','DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
  .select(['VV','VH']);
var sar = s1.median();
var difference = sar.select('VV').subtract(sar.select('VH')).rename('VV_minus_VH');
var elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elev');
var terrain = ee.Terrain.products(elevation);
var aspect = terrain.select('aspect').multiply(Math.PI/180);
var image = reflectance.addBands(ndvi).addBands(sar).addBands(difference)
  .addBands(elevation).addBands(terrain.select('slope'))
  .addBands(aspect.sin().rename('aspect_sin'))
  .addBands(aspect.cos().rename('aspect_cos'))
  .select(RASTER).reproject(PROJECTION);

// Missing neighbours must count as zero coverage, rather than disappearing.
var valid = image.mask().reduce(ee.Reducer.min())
  .unmask(0,false).gt(0).rename('center_valid').reproject(PROJECTION);
var kernel = ee.Kernel.square({radius:8,units:'pixels',normalize:false});
var fraction = valid.reduceNeighborhood({reducer:ee.Reducer.mean(),
  kernel:kernel,inputWeight:'kernel',skipMasked:false})
  .rename('patch_valid_fraction');
var complete = valid.reduceNeighborhood({reducer:ee.Reducer.min(),
  kernel:kernel,inputWeight:'kernel',skipMasked:false})
  .rename('patch_all_valid');
var counts = s2.select('B4').count().unmask(0,false).rename('n_s2_clear')
  .addBands(s1.select('VV').count().unmask(0,false).rename('n_s1'));
var diagnostics = valid.addBands(fraction).addBands(complete).addBands(counts);
var table = diagnostics.reduceRegions({collection:points,
  reducer:ee.Reducer.first(),crs:PROJECTION,tileScale:4});
table = image.reduceRegions({collection:table,reducer:ee.Reducer.first(),
  crs:PROJECTION,tileScale:4});

// The three climate covariates remain on their native ERA5-Land grid.
var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
var eraProjection = ee.Image(era.first()).select('temperature_2m').projection();
var climate = ee.Image.cat([
  era.filterDate('2017-12-01','2018-03-01').select('total_precipitation_sum')
    .sum().rename('tp_winter_m'),
  era.filterDate('2018-06-01','2018-09-01').select('temperature_2m')
    .mean().subtract(273.15).rename('t2m_summer_C'),
  era.filterDate('2018-01-01','2019-01-01').select('volumetric_soil_water_layer_1')
    .mean().rename('sm_annual'),
  ee.Image.pixelCoordinates(eraProjection).rename(['era_grid_x','era_grid_y'])
]);
table = climate.reduceRegions({collection:table,reducer:ee.Reducer.first(),
  crs:eraProjection,tileScale:4})
  .map(function(f){return f.set('predictor_year',2018);});

Export.table.toDrive({collection:table,
  description:'SOC_ES_2018_coverage',folder:'SOC_Spain_replication',
  fileNamePrefix:'SOC_ES_2018_coverage',fileFormat:'CSV',
  selectors:['point_id','POINTID','NUTS_0','OC','Depth','SURVEY_DATE',
    'LC','LC0_Desc','LC1_Desc','source_lon','source_lat',
    'source_x3035','source_y3035','predictor_year',
    'center_valid','patch_valid_fraction','patch_all_valid','n_s2_clear','n_s1']
    .concat(RASTER).concat(['tp_winter_m','t2m_summer_C','sm_annual',
      'era_grid_x','era_grid_y'])});
print('NEXT: Tasks > SOC_ES_2018_coverage > Run. Send the completed CSV.');
Map.setCenter(-3.5,40,5);
